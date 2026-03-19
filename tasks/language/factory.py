import os
import sys
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings

load_dotenv()

_llm_cache: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    """
    Returns a ChatModel based on LLM_PROVIDER.
    Cached after first call — reuses the same instance.
    """
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache

    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    temp = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    print(f"🏭 Factory: Initializing LLM -> {provider.upper()} (Temp: {temp})")

    try:
        # PATH 1: Native Azure (Best for GPT-4, o1)
        if provider == "azure_native":
            import httpx
            from langchain_openai import AzureChatOpenAI
            deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
            kwargs = dict(
                azure_deployment=deployment,
                api_version=os.getenv("AZURE_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_API_BASE"),
                api_key=os.getenv("AZURE_API_KEY"),
                temperature=temp,
                http_client=httpx.Client(verify=False),
                http_async_client=httpx.AsyncClient(verify=False),
            )
            # o-series models support reasoning_effort for speed control
            if deployment and deployment.startswith("o"):
                kwargs["reasoning_effort"] = os.getenv("AZURE_REASONING_EFFORT", "low")
            _llm_cache = AzureChatOpenAI(**kwargs)

        # PATH 2: OpenAI Compatible (The Universal Adapter)
        elif provider == "openai_compatible":
            from langchain_openai import ChatOpenAI

            base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
            api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
            model = os.getenv("OPENAI_COMPATIBLE_MODEL")

            if not all([base_url, api_key, model]):
                raise ValueError("❌ Provider is 'openai_compatible' but BASE_URL, API_KEY, or MODEL is missing.")

            _llm_cache = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=temp
            )

        # PATH 3: Local Offline
        elif provider == "ollama":
            from langchain_community.chat_models import ChatOllama
            _llm_cache = ChatOllama(
                model=os.getenv("OLLAMA_MODEL", "llama3"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=temp
            )

        else:
            raise ValueError(f"❌ Unsupported LLM_PROVIDER: {provider}")

    except ImportError as e:
        print(f"❌ Dependency Error: {e}")
        sys.exit(1)

    return _llm_cache


def get_embeddings() -> Embeddings:
    """
    Returns an Embedding model.
    Usually, we stick to Azure Ada-002 (Native) or Offline HuggingFace.
    """
    provider = os.getenv("EMBEDDING_PROVIDER", "azure_native").lower()

    print(f"🏭 Factory: Initializing Embeddings -> {provider.upper()}")

    if provider == "azure_native":
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_key=os.getenv("AZURE_API_KEY"),
        )

    elif provider == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    else:
        raise ValueError(f"❌ Unsupported Embedding Provider: {provider}")
