import os
import joblib

from src.config import Config
from src.data_loader import load_data
from src.preprocess import split_features_label, scale_features
from src.model import build_model
from src.train import split_train_test
from src.evaluate import evaluate_model

def main():
    config = Config()

    os.makedirs("artifacts", exist_ok=True)

    df = load_data(config.data_path)

    X, y = split_features_label(df, config.label_column)
    X_scaled, scaler = scale_features(X)

    X_train, X_test, y_train, y_test = split_train_test(X_scaled, y)

    model = build_model()
    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    joblib.dump(model, config.model_path)
    joblib.dump(scaler, config.scaler_path)

    print("\nModel and scaler saved successfully.")

if __name__ == "__main__":
    main()
