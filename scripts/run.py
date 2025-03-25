import os
import glob
import subprocess
from pathlib import Path
import json

# Define directories
# data_dir = Path("/home/nmearl/research/tde_agn_comparison/data")
templates_dir = Path.home() / Path("research/tde_agn_comparison/tde_templates")
results_dir = Path.home() / Path("research/tde_agn_comparison/tde_results")

# Get all files in the data and templates directories
# data_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
template_files = sorted(glob.glob(os.path.join(templates_dir, "*.json")))

# Ensure matching filenames exist in both directories
# for data_file in sorted(data_dir.glob("*.csv")):
#     base_name = data_file.stem
#     template_file = os.path.join(templates_dir, f"{base_name}.json")
for template_file in sorted(templates_dir.glob("*.json")):
    base_name = template_file.stem

    with open(template_file, "r") as f:
        template = json.load(f)

    data_file = template["data_path"]

    if os.path.exists(data_file):
        output_dir = os.path.join(results_dir, base_name)
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

        # Construct the command
        cmd = [
            "feadme",
            data_file,
            str(template_file),
            "--output-dir",
            output_dir,
            "--num_warmup=10000",
            "--num_samples=10000",
            "--num_chains=1",
        ]

        # Run the command
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        print(f"Warning: No matching template found for {data_file}")
