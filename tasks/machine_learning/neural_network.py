"""
tasks/machine_learning/neural_network.py
-----------------------------------------
PyTorch feedforward neural network for regression with Optuna hyperparameter
optimisation (Bayesian / TPE search).

Design goals:
  - Mirror the fit / predict / evaluate / save / load interface of TabularPipeline
  - Fully configurable architecture: depth, width, activation, dropout, etc.
  - One-line HPO: call .optimise() instead of .fit() to let Optuna find the best
    hyperparameters before training the final model
  - Graceful degradation: falls back to CPU if CUDA is unavailable

Usage (manual config):
    from tasks.machine_learning.neural_network import NeuralNetRegressor

    model = NeuralNetRegressor(hidden_layers=[128, 64], lr=1e-3)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = model.evaluate(X_val, y_val)
    model.save("models/nn_v1.pt")

    # Load later:
    model = NeuralNetRegressor.load("models/nn_v1.pt")

Usage (Optuna HPO):
    model = NeuralNetRegressor()
    model.optimise(X_train, y_train, n_trials=50)
    metrics = model.evaluate(X_val, y_val)
    model.save("models/nn_best.pt")
"""

from __future__ import annotations

import os
import warnings
import time
from typing import Callable, Literal, Optional

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional dependencies — graceful degradation
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# ---------------------------------------------------------------------------
# Supported choices (used both internally and by Optuna search space)
# ---------------------------------------------------------------------------
ACTIVATIONS: dict[str, Callable] = {}
if _HAS_TORCH:
    ACTIVATIONS = {
        "relu":     nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu":      nn.ELU,
        "gelu":     nn.GELU,
        "tanh":     nn.Tanh,
        "silu":     nn.SiLU,
    }

LOSS_FUNCTIONS: dict[str, Callable] = {}
if _HAS_TORCH:
    LOSS_FUNCTIONS = {
        "mse":   nn.MSELoss,
        "mae":   nn.L1Loss,
        "huber": nn.HuberLoss,
        "log_cosh": None,   # custom — built below
    }

OPTIMIZERS: dict[str, type] = {}
if _HAS_TORCH:
    OPTIMIZERS = {
        "adam":     torch.optim.Adam,
        "adamw":    torch.optim.AdamW,
        "sgd":      torch.optim.SGD,
        "rmsprop":  torch.optim.RMSprop,
    }


# ---------------------------------------------------------------------------
# Custom loss: Log-Cosh  (smooth, robust to outliers like Huber but differentiable)
# ---------------------------------------------------------------------------
def _log_cosh_loss(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    diff = pred - target
    return torch.mean(torch.log(torch.cosh(diff + 1e-12)))


# ---------------------------------------------------------------------------
# Network definition
# ---------------------------------------------------------------------------
def _build_network(
    input_dim: int,
    hidden_layers: list[int],
    activation: str,
    dropout_rate: float,
    use_batch_norm: bool,
) -> "nn.Sequential":
    """Construct a fully-connected feed-forward network (Linear → BN? → Act → Drop)."""
    assert _HAS_TORCH, "PyTorch is required. Install it via: pip install torch"

    layers: list[nn.Module] = []
    in_dim = input_dim
    act_cls = ACTIVATIONS[activation]

    for out_dim in hidden_layers:
        layers.append(nn.Linear(in_dim, out_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(act_cls())
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        in_dim = out_dim

    layers.append(nn.Linear(in_dim, 1))   # regression output (scalar)
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class NeuralNetRegressor:
    """
    Feedforward neural network regressor built on PyTorch.

    Hyperparameters (set at construction or found automatically via .optimise()):
        hidden_layers   : List of layer widths, e.g. [256, 128, 64]
        lr              : Learning rate
        weight_decay    : L2 regularisation penalty
        dropout_rate    : Dropout probability applied after each hidden layer
        activation      : One of relu / leaky_relu / elu / gelu / tanh / silu
        loss_fn         : One of mse / mae / huber / log_cosh
        optimizer       : One of adam / adamw / sgd / rmsprop
        batch_size      : Mini-batch size during training
        max_epochs      : Maximum number of training epochs
        patience        : Early stopping — stop after this many epochs without
                           improvement on the validation loss
        val_fraction    : Fraction of training data held out as validation set
                          (used for early stopping; ignored when val data is
                          provided explicitly to .fit())
        use_batch_norm  : Insert BatchNorm1d after each linear layer
        scheduler       : LR scheduler — none / cosine / step / reduce_on_plateau
        scheduler_step_size : Step size for StepLR (epochs)
        scheduler_gamma : Decay factor for StepLR
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        hidden_layers: list[int] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        loss_fn: str = "mse",
        optimizer: str = "adam",
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 20,
        val_fraction: float = 0.1,
        use_batch_norm: bool = True,
        scheduler: Literal["none", "cosine", "step", "reduce_on_plateau"] = "cosine",
        scheduler_step_size: int = 50,
        scheduler_gamma: float = 0.5,
        device: str | None = None,
        verbose: bool = True,
    ):
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is not installed. "
                "Run:  pip install torch  (or pip install -e '.[vision]')"
            )

        self.hidden_layers = hidden_layers if hidden_layers is not None else [128, 64]
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.use_batch_norm = use_batch_norm
        self.scheduler = scheduler
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.verbose = verbose

        # Resolve device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # State filled after fit()
        self._network: Optional[nn.Sequential] = None
        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        self._input_dim: int = 0
        self._feature_names: list[str] = []
        self._training_history: dict = {"train_loss": [], "val_loss": []}
        self._best_params: dict = {}    # populated by .optimise()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> "NeuralNetRegressor":
        """
        Train the network on (X, y).

        Args:
            X:     Feature matrix
            y:     Continuous target vector
            X_val: Optional held-out validation features (overrides val_fraction split)
            y_val: Optional held-out validation targets

        Returns:
            self (chainable)
        """
        X_np, y_np = self._to_numpy(X, y)
        self._feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else \
            [f"f{i}" for i in range(X_np.shape[1])]
        self._input_dim = X_np.shape[1]

        # Scale features and (importantly) the target — helps gradient flow
        X_np = self._scaler_X.fit_transform(X_np)
        y_np = self._scaler_y.fit_transform(y_np.reshape(-1, 1)).ravel()

        # Validation set
        if X_val is not None and y_val is not None:
            X_val_np, y_val_np = self._to_numpy(X_val, y_val)
            X_val_np = self._scaler_X.transform(X_val_np)
            y_val_np = self._scaler_y.transform(y_val_np.reshape(-1, 1)).ravel()
        else:
            X_np, X_val_np, y_np, y_val_np = train_test_split(
                X_np, y_np,
                test_size=self.val_fraction,
                random_state=RANDOM_SEED,
            )

        # Build model
        self._network = _build_network(
            input_dim=self._input_dim,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm,
        ).to(self.device)

        # Initialise weights (Kaiming for ReLU-family, Xavier otherwise)
        self._init_weights()

        # Loss function
        loss_fn = self._get_loss_fn()

        # Optimiser
        opt_cls = OPTIMIZERS[self.optimizer]
        opt_kwargs: dict = {"lr": self.lr, "weight_decay": self.weight_decay}
        if self.optimizer == "sgd":
            opt_kwargs["momentum"] = 0.9
        optimiser = opt_cls(self._network.parameters(), **opt_kwargs)

        # LR scheduler
        sched = self._build_scheduler(optimiser)

        # DataLoaders
        train_loader = self._make_loader(X_np, y_np, shuffle=True)
        val_loader   = self._make_loader(X_val_np, y_val_np, shuffle=False)

        # Training loop with early stopping
        best_val_loss = np.inf
        best_state = None
        epochs_no_improve = 0
        self._training_history = {"train_loss": [], "val_loss": []}

        t0 = time.time()
        if self.verbose:
            arch_str = " → ".join(str(w) for w in self.hidden_layers)
            print(f"\n🧠 NeuralNetRegressor  device={self.device}  arch=[{arch_str}]")
            print(f"   lr={self.lr}  loss={self.loss_fn}  opt={self.optimizer}"
                  f"  batch={self.batch_size}  max_epochs={self.max_epochs}\n")

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch(train_loader, loss_fn, optimiser)
            val_loss   = self._eval_epoch(val_loader, loss_fn)

            self._training_history["train_loss"].append(train_loss)
            self._training_history["val_loss"].append(val_loss)

            # Scheduler step
            if sched is not None:
                if self.scheduler == "reduce_on_plateau":
                    sched.step(val_loss)
                else:
                    sched.step()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self._network.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if self.verbose and epoch % max(1, self.max_epochs // 10) == 0:
                print(f"   Epoch {epoch:>4d}/{self.max_epochs}  "
                      f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")

            if epochs_no_improve >= self.patience:
                if self.verbose:
                    print(f"   ⏹  Early stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs)")
                break

        # Restore best weights
        if best_state is not None:
            self._network.load_state_dict(best_state)

        elapsed = time.time() - t0
        if self.verbose:
            print(f"\n✅ Training complete — best val_loss={best_val_loss:.5f}  ({elapsed:.1f}s)")

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return continuous predictions (inverse-scaled to original target space)."""
        self._assert_fitted()
        X_np = self._to_numpy(X)
        X_np = self._scaler_X.transform(X_np)

        self._network.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_np, dtype=torch.float32, device=self.device)
            preds_scaled = self._network(X_t).cpu().numpy().ravel()

        return self._scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

    def evaluate(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> dict:
        """Compute regression metrics on a held-out set and print them."""
        self._assert_fitted()
        y_np = np.asarray(y).ravel()
        preds = self.predict(X)

        mae  = mean_absolute_error(y_np, preds)
        rmse = np.sqrt(mean_squared_error(y_np, preds))
        r2   = r2_score(y_np, preds)
        metrics = {
            "mae":  round(float(mae),  4),
            "rmse": round(float(rmse), 4),
            "r2":   round(float(r2),   4),
        }
        print("📊 Evaluation:", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Hyperparameter optimisation with Optuna
    # ------------------------------------------------------------------

    def optimise(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        n_trials: int = 50,
        timeout: float | None = None,
        search_space: dict | None = None,
        pruning: bool = True,
    ) -> "NeuralNetRegressor":
        """
        Use Optuna (Bayesian / TPE) to search the hyperparameter space, then
        refit the best configuration on all training data.

        Args:
            X:            Training features
            y:            Continuous targets
            n_trials:     Number of Optuna trials
            timeout:      Optional wall-clock limit in seconds
            search_space: Override default search space boundaries (dict of
                          param_name -> (low, high) or list of choices).
                          Keys: depth, min_width, max_width, lr, weight_decay,
                                dropout_rate, batch_size, activation, loss_fn,
                                optimizer, use_batch_norm, scheduler
            pruning:      Enable Optuna's MedianPruner for slow trials

        Returns:
            self (with best params set and final model trained on full data)
        """
        if not _HAS_OPTUNA:
            raise ImportError(
                "Optuna is not installed. Run:  pip install optuna"
            )

        ss = self._default_search_space()
        if search_space:
            ss.update(search_space)

        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10) \
            if pruning else optuna.pruners.NopPruner()
        sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)

        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
            study_name="nn_regression_hpo",
        )

        # Inner objective — trains a smaller version for HPO speed
        X_tr, X_val, y_tr, y_val = train_test_split(
            *self._to_numpy(X, y),
            test_size=max(self.val_fraction, 0.15),
            random_state=RANDOM_SEED,
        )

        def objective(trial: optuna.Trial) -> float:
            # ---- Architecture ----
            depth = trial.suggest_int("depth", *ss["depth"])
            width_choices = ss.get("width_choices")
            if width_choices:
                widths = [trial.suggest_categorical(f"width_{i}", width_choices)
                          for i in range(depth)]
            else:
                widths = [trial.suggest_int(f"width_{i}", *ss["width"])
                          for i in range(depth)]

            # ---- Regularisation ----
            dropout  = trial.suggest_float("dropout_rate", *ss["dropout_rate"])
            wd       = trial.suggest_float("weight_decay", *ss["weight_decay"], log=True)
            use_bn   = trial.suggest_categorical("use_batch_norm", ss["use_batch_norm"])

            # ---- Optimisation ----
            lr       = trial.suggest_float("lr", *ss["lr"], log=True)
            opt      = trial.suggest_categorical("optimizer", ss["optimizer"])
            bs       = trial.suggest_categorical("batch_size", ss["batch_size"])
            act      = trial.suggest_categorical("activation", ss["activation"])
            loss     = trial.suggest_categorical("loss_fn", ss["loss_fn"])
            sched    = trial.suggest_categorical("scheduler", ss["scheduler"])

            # Build a temporary regressor with these params
            tmp = NeuralNetRegressor(
                hidden_layers=widths,
                lr=lr,
                weight_decay=wd,
                dropout_rate=dropout,
                activation=act,
                loss_fn=loss,
                optimizer=opt,
                batch_size=int(bs),
                max_epochs=min(self.max_epochs, 100),   # cap epochs during HPO
                patience=min(self.patience, 15),
                use_batch_norm=bool(use_bn),
                scheduler=sched,
                device=str(self.device),
                verbose=False,
            )
            tmp.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)

            preds = tmp.predict(X_val)
            return float(np.sqrt(mean_squared_error(y_val, preds)))

        if self.verbose:
            print(f"\n🔍 Optuna search — n_trials={n_trials}  sampler=TPE  device={self.device}")

        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=self.verbose)

        best = study.best_params
        if self.verbose:
            print(f"\n🏆 Best trial  RMSE={study.best_value:.5f}")
            print(f"   Params: {best}")

        # Reconstruct best architecture
        depth = best["depth"]
        width_choices = ss.get("width_choices")
        if width_choices:
            best_widths = [best[f"width_{i}"] for i in range(depth)]
        else:
            best_widths = [best[f"width_{i}"] for i in range(depth)]

        # Apply best params to self
        self.hidden_layers   = best_widths
        self.lr              = best["lr"]
        self.weight_decay    = best["weight_decay"]
        self.dropout_rate    = best["dropout_rate"]
        self.activation      = best["activation"]
        self.loss_fn         = best["loss_fn"]
        self.optimizer       = best["optimizer"]
        self.batch_size      = int(best["batch_size"])
        self.use_batch_norm  = bool(best["use_batch_norm"])
        self.scheduler       = best["scheduler"]
        self._best_params    = dict(best)

        # Final fit on all data
        if self.verbose:
            print("\n🔁 Refitting on full training data with best hyperparameters …")
        self.fit(X, y)

        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save model weights, scalers, and config to disk.
        State is stored as a plain dict (no pickle of nn.Module) for
        forward-compatibility with different PyTorch versions.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "config": self._get_config(),
            "network_state_dict": self._network.state_dict() if self._network else None,
            "scaler_X": self._scaler_X,
            "scaler_y": self._scaler_y,
            "input_dim": self._input_dim,
            "feature_names": self._feature_names,
            "training_history": self._training_history,
            "best_params": self._best_params,
        }
        joblib.dump(state, path)
        print(f"💾 NeuralNetRegressor saved → {path}")

    @classmethod
    def load(cls, path: str) -> "NeuralNetRegressor":
        """Load a previously saved NeuralNetRegressor."""
        state = joblib.load(path)
        cfg   = state["config"]
        obj   = cls(**{k: v for k, v in cfg.items() if k != "device"})
        obj.device        = torch.device(cfg["device"])
        obj._scaler_X     = state["scaler_X"]
        obj._scaler_y     = state["scaler_y"]
        obj._input_dim    = state["input_dim"]
        obj._feature_names = state["feature_names"]
        obj._training_history = state["training_history"]
        obj._best_params  = state["best_params"]

        if state["network_state_dict"] is not None:
            obj._network = _build_network(
                input_dim=obj._input_dim,
                hidden_layers=obj.hidden_layers,
                activation=obj.activation,
                dropout_rate=obj.dropout_rate,
                use_batch_norm=obj.use_batch_norm,
            ).to(obj.device)
            obj._network.load_state_dict(state["network_state_dict"])
            obj._network.eval()

        print(f"📂 NeuralNetRegressor loaded ← {path}")
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_config(self) -> dict:
        return {
            "hidden_layers":       self.hidden_layers,
            "lr":                  self.lr,
            "weight_decay":        self.weight_decay,
            "dropout_rate":        self.dropout_rate,
            "activation":          self.activation,
            "loss_fn":             self.loss_fn,
            "optimizer":           self.optimizer,
            "batch_size":          self.batch_size,
            "max_epochs":          self.max_epochs,
            "patience":            self.patience,
            "val_fraction":        self.val_fraction,
            "use_batch_norm":      self.use_batch_norm,
            "scheduler":           self.scheduler,
            "scheduler_step_size": self.scheduler_step_size,
            "scheduler_gamma":     self.scheduler_gamma,
            "device":              str(self.device),
            "verbose":             self.verbose,
        }

    @staticmethod
    def _default_search_space() -> dict:
        return {
            # Architecture
            "depth":         (1, 5),                    # number of hidden layers
            "width":         (32, 512),                 # neurons per hidden layer
            # Regularisation
            "dropout_rate":  (0.0, 0.5),
            "weight_decay":  (1e-6, 1e-2),
            "use_batch_norm": [True, False],
            # Optimisation
            "lr":            (1e-4, 1e-2),
            "optimizer":     ["adam", "adamw", "sgd", "rmsprop"],
            "batch_size":    [32, 64, 128, 256],
            "activation":    ["relu", "leaky_relu", "elu", "gelu", "silu"],
            "loss_fn":       ["mse", "mae", "huber", "log_cosh"],
            "scheduler":     ["none", "cosine", "step", "reduce_on_plateau"],
        }

    def _get_loss_fn(self) -> Callable:
        if self.loss_fn == "log_cosh":
            return _log_cosh_loss
        cls = LOSS_FUNCTIONS[self.loss_fn]
        return cls()

    def _build_scheduler(self, optimiser):
        if self.scheduler == "none":
            return None
        if self.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=self.max_epochs)
        if self.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimiser,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
            )
        if self.scheduler == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser, patience=self.patience // 2, factor=0.5)
        raise ValueError(f"Unknown scheduler: '{self.scheduler}'")

    def _init_weights(self):
        relu_like = {"relu", "leaky_relu", "elu", "silu"}
        for module in self._network.modules():
            if isinstance(module, nn.Linear):
                if self.activation in relu_like:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X_t, y_t)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    def _train_epoch(
        self,
        loader: DataLoader,
        loss_fn: Callable,
        optimiser,
    ) -> float:
        self._network.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            optimiser.zero_grad()
            preds = self._network(X_batch)
            loss  = loss_fn(preds, y_batch)
            loss.backward()
            # Gradient clipping — prevents exploding gradients in deep nets
            nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
            optimiser.step()
            total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    def _eval_epoch(self, loader: DataLoader, loss_fn: Callable) -> float:
        self._network.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                preds = self._network(X_batch)
                loss  = loss_fn(preds, y_batch)
                total_loss += loss.item() * len(X_batch)
        return total_loss / len(loader.dataset)

    @staticmethod
    def _to_numpy(
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
    ):
        X_np = X.values.astype(np.float32) if isinstance(X, pd.DataFrame) else \
            np.asarray(X, dtype=np.float32)
        if y is None:
            return X_np
        y_np = y.values.astype(np.float32) if isinstance(y, pd.Series) else \
            np.asarray(y, dtype=np.float32)
        return X_np, y_np

    def _assert_fitted(self):
        if self._network is None:
            raise RuntimeError("Model is not fitted. Call .fit() or .optimise() first.")


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    X, y = data.data, data.target

    split = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # --- Manual configuration ---
    print("=" * 60)
    print("1. Manual configuration smoke test")
    print("=" * 60)
    model = NeuralNetRegressor(
        hidden_layers=[128, 64, 32],
        lr=1e-3,
        loss_fn="huber",
        max_epochs=100,
        patience=15,
        verbose=True,
    )
    model.fit(X_train, y_train)
    model.evaluate(X_test, y_test)
    model.save("models/nn_manual_smoke.pt")

    # --- Optuna HPO ---
    print("\n" + "=" * 60)
    print("2. Optuna HPO smoke test (10 trials)")
    print("=" * 60)
    model_hpo = NeuralNetRegressor(max_epochs=80, patience=10, verbose=True)
    model_hpo.optimise(X_train, y_train, n_trials=10)
    model_hpo.evaluate(X_test, y_test)
    model_hpo.save("models/nn_hpo_smoke.pt")

    # --- Round-trip load ---
    loaded = NeuralNetRegressor.load("models/nn_hpo_smoke.pt")
    loaded.evaluate(X_test, y_test)
