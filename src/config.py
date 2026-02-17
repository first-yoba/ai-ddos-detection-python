from dataclasses import dataclass

@dataclass
class Config:
    data_path: str = "data/sample_traffic.csv"
    label_column: str = "label"
    model_path: str = "artifacts/ddos_model.joblib"
    scaler_path: str = "artifacts/scaler.joblib"
