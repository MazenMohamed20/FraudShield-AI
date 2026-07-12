import os
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
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


# Build Neural Network

def build_model(input_dim):

    model = Sequential([

        Input(shape=(input_dim,)),

        Dense(64, activation="relu"),

        Dropout(0.3),

        Dense(32, activation="relu"),

        Dropout(0.3),

        Dense(1, activation="sigmoid")

    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# Train Neural Network

def train_model(model, X_train, y_train):

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=256,
        callbacks=[early_stop],
        verbose=1
    )

    return history

# Prediction

def predict(model, X_test):

    # Probability of Fraud
    y_prob = model.predict(X_test, verbose=0).flatten()

    # Convert Probability to Class (0 or 1)
    y_pred = (y_prob >= 0.5).astype(int)

    return y_pred, y_prob


# Model Evaluation

def evaluate_model(y_test, y_pred, y_prob):

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("=" * 50)
    print("Neural Network Evaluation")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {roc_auc:.4f}")

    print("=" * 50)
    print("Classification Report")
    print(classification_report(y_test, y_pred, zero_division=0))

    return accuracy, precision, recall, f1, roc_auc


# Confusion Matrix

def plot_confusion_matrix(y_test, y_pred):

    figures_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\reports\figures"

    os.makedirs(figures_path, exist_ok=True)

    plt.figure(figsize=(6,6))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="Blues"
    )

    plt.title("Neural Network Confusion Matrix")

    plt.savefig(
        os.path.join(
            figures_path,
            "neural_network_confusion_matrix.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ROC Curve

def plot_roc_curve(y_test, y_prob):

    figures_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\reports\figures"

    os.makedirs(figures_path, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8,6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {auc:.4f}"
    )

    plt.plot([0,1],[0,1],"--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("Neural Network ROC Curve")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        os.path.join(
            figures_path,
            "neural_network_roc_curve.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# Plot Training History

def plot_training_history(history):

    figures_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\reports\figures"

    os.makedirs(figures_path, exist_ok=True)

    # Accuracy
    plt.figure(figsize=(8,6))

    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Neural Network Accuracy")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(
            figures_path,
            "neural_network_accuracy.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # Loss
    plt.figure(figsize=(8,6))

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Neural Network Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        os.path.join(
            figures_path,
            "neural_network_loss.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# Save Model

def save_model(model):

    models_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\models"

    os.makedirs(models_path, exist_ok=True)

    model.save(
        os.path.join(
            models_path,
            "neural_network.keras"
        )
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
        os.path.join(
            metrics_path,
            "neural_network_metrics.txt"
        ),
        "w"
    ) as file:

        file.write("Neural Network Results\n")
        file.write("=" * 40 + "\n")
        file.write(f"Accuracy : {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall   : {recall:.4f}\n")
        file.write(f"F1 Score : {f1:.4f}\n")
        file.write(f"ROC AUC  : {roc_auc:.4f}\n")

    print("=" * 50)
    print("Metrics Saved Successfully!")
    print("=" * 50)


# Main

def main():

    print("=" * 60)
    print("FraudShield AI - Neural Network")
    print("=" * 60)

    # Load Data
    X_train, X_test, y_train, y_test = load_processed_data()

    # Build Model
    model = build_model(X_train.shape[1])

    # Train
    history = train_model(
        model,
        X_train,
        y_train
    )

    # Plot History
    plot_training_history(history)

    # Prediction
    y_pred, y_prob = predict(
        model,
        X_test
    )

    # Evaluation
    accuracy, precision, recall, f1, roc_auc = evaluate_model(
        y_test,
        y_pred,
        y_prob
    )

    # Confusion Matrix
    plot_confusion_matrix(
        y_test,
        y_pred
    )

    # ROC Curve
    plot_roc_curve(
        y_test,
        y_prob
    )

    # Save Model
    save_model(model)

    # Save Metrics
    save_metrics(
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    )

    print("=" * 60)
    print("Neural Network Completed Successfully!")
    print("=" * 60)


# Run

if __name__ == "__main__":
    main()