import argparse
import os
import pandas as pd
import sweetviz as sv
import sys

def generate_report(file_path: str, output_dir: str = "notebooks/reports"):
    """Reads a dataset and generates a blazingly fast HTML profiling report using Sweetviz."""
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.basename(file_path)
    base_name, ext = os.path.splitext(file_name)

    print(f"📊 Loading dataset: {file_name}...")
    
    try:
        if ext.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif ext.lower() in['.xls', '.xlsx']:
            df = pd.read_excel(file_path)
        elif ext.lower() == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            print(f"❌ Error: Unsupported file extension '{ext}'. Use CSV, Excel, or Parquet.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        sys.exit(1)

    print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    print(f"⚙️ Generating Sweetviz profile report...")

    # Generate the report. Sweetviz is highly optimized for this.
    report = sv.analyze(df)
    
    output_path = os.path.join(output_dir, f"{base_name}_report.html")
    # Save the report without attempting to open it in a WSL-headless browser
    report.show_html(filepath=output_path, open_browser=False)
    
    print(f"🚀 Success! Report saved to: {output_path}")
    print(f"Open this file in your Windows browser to view the analysis.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an EDA report using Sweetviz.")
    parser.add_argument("file_path", type=str, help="Path to the dataset (CSV, Excel, Parquet)")
    parser.add_argument("--output", type=str, default="notebooks/reports", help="Output directory for the HTML report")
    
    args = parser.parse_args()
    generate_report(args.file_path, args.output)