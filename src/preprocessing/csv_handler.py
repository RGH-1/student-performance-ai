from pathlib import Path
import pandas as pd

def load_data(file_name="original"):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    if(file_name == "original"):
        file_path = PROJECT_ROOT / "data/raw/Student_Performance_Dataset.csv"
    else:
        file_path = PROJECT_ROOT / "data/processed" / file_name
    file_path = file_path.resolve()

    students = pd.read_csv(file_path, index_col="student_id")
    return students


def save_data(df: pd.DataFrame, file_name):
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    file_path = PROJECT_ROOT / "data/processed" / file_name
    df.to_csv(file_path)