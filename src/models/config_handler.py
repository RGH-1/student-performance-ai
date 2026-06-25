from pathlib import Path
import json

def get_grid_config(model_name):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    if (model_name == "random_forest"):
        file_path = PROJECT_ROOT / "configs/rf_grid_config.json"
    elif (model_name == "gradient_boost"):
        file_path = PROJECT_ROOT / "configs/gb_grid_config.json"
    else:
        raise ValueError("Use a valid model name")
    file_path = file_path.resolve()

    with open(file_path, "r") as f:
        config = json.load(f)
    
    return config