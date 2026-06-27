import pickle
from pathlib import Path

def get_model_path(model_type):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    type = None
    if(model_type == "ml"):
        type = "ml_models"
    elif(model_type == "dl"):
        type = "dl_models"
    else:
        raise ValueError("No model type named:", model_type)

    file_path = PROJECT_ROOT / "models" / type
    return file_path


def load_ml_model(model_file_name):
    file_path = get_model_path('ml') / model_file_name

    with open(file_path, 'rb') as m:
        model = pickle.load(m)
    
    return model

def save_ml_model(model, model_name):
    model_name = model_name + '.pkl'

    file_path = get_model_path('ml') / model_name

    with open(file_path, 'wb') as f:
        pickle.dump(model, f)