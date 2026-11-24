from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normale", "Fraude"],
        zero_division=0
    ))

    print("ROC-AUC :", roc_auc_score(y_test, y_proba))
    print("Average Precision :", average_precision_score(y_test, y_proba))