import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve
)

# Load Processed Data

def load_processed_data():

    processed_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\data\processed"

    X_train = pd.read_csv(os.path.join(processed_path, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_path, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(processed_path, "y_test.csv")).squeeze()

    print("=" * 50)
    print("Processed Data Loaded Successfully!")
    print(f"X_train Shape : {X_train.shape}")
    print(f"X_test Shape  : {X_test.shape}")
    print(f"y_train Shape : {y_train.shape}")
    print(f"y_test Shape  : {y_test.shape}")
    print("=" * 50)

    return X_train, X_test, y_train, y_test


# Train Decision Tree

def train_model(X_train, y_train):

    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5
    )

    model.fit(X_train, y_train)

    print("=" * 50)
    print("Decision Tree Trained Successfully!")
    print("=" * 50)

    return model
# Prediction

def predict(model, X_test):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    return y_pred, y_prob


# Model Evaluation

def evaluate_model(y_test, y_pred, y_prob):

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("=" * 50)
    print("Decision Tree Evaluation")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    print("=" * 50)
    print("Classification Report")
    print(classification_report(y_test, y_pred))

    return accuracy, precision, recall, f1, roc_auc

# Plot Confusion Matrix

def plot_confusion_matrix(y_test, y_pred):

    reports_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\reports\figures"

    os.makedirs(reports_path, exist_ok=True)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="Blues"
    )

    plt.title("Decision Tree - Confusion Matrix")

    plt.savefig(
        os.path.join(reports_path, "decision_tree_confusion_matrix.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# Plot ROC Curve

def plot_roc_curve(y_test, y_prob):

    reports_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\reports\figures"

    os.makedirs(reports_path, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {auc:.4f}"
    )

    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(reports_path, "decision_tree_roc_curve.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# Save Model

def save_model(model):

    models_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\models"

    os.makedirs(models_path, exist_ok=True)

    joblib.dump(
        model,
        os.path.join(models_path, "decision_tree.pkl")
    )

    print("=" * 50)
    print("Model Saved Successfully!")
    print("=" * 50)


# Save Metrics

def save_metrics(
    accuracy,
    precision,
    recall,
    f1,
    roc_auc
):

    metrics_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\reports\metrics"

    os.makedirs(metrics_path, exist_ok=True)

    with open(
        os.path.join(metrics_path, "decision_tree_metrics.txt"),
        "w"
    ) as file:

        file.write("Decision Tree Results\n")
        file.write("=" * 40 + "\n")
        file.write(f"Accuracy : {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall   : {recall:.4f}\n")
        file.write(f"F1 Score : {f1:.4f}\n")
        file.write(f"ROC AUC  : {roc_auc:.4f}\n")

    print("=" * 50)
    print("Metrics Saved Successfully!")
    print("=" * 50)


# Main Function

def main():

    X_train, X_test, y_train, y_test = load_processed_data()

    model = train_model(
        X_train,
        y_train
    )

    y_pred, y_prob = predict(
        model,
        X_test
    )

    accuracy, precision, recall, f1, roc_auc = evaluate_model(
        y_test,
        y_pred,
        y_prob
    )

    plot_confusion_matrix(
        y_test,
        y_pred
    )

    plot_roc_curve(
        y_test,
        y_prob
    )

    save_model(
        model
    )

    save_metrics(
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    )


# Run

if __name__ == "__main__":

    main()