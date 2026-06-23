from pathlib import Path
import pandas as pd

def load_data():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    file_path = PROJECT_ROOT / "data/raw/Student_Performance_Dataset.csv"
    file_path = file_path.resolve()

    students = pd.read_csv(file_path, index_col="student_id")
    return students