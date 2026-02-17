from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    # ROC-AUC needs predicted probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        print(f"\nROC-AUC: {auc:.4f}")
