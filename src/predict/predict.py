import os
import joblib
import pandas as pd

# Load Random Forest Model

def load_model():

    model_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\models\random_forest.pkl"

    model = joblib.load(model_path)

    print("=" * 50)
    print("Random Forest Model Loaded Successfully!")
    print("=" * 50)

    return 

# Load Robust Scaler

def load_scaler():

    scaler_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\models\robust_scaler.pkl"

    scaler = joblib.load(scaler_path)

    print("=" * 50)
    print("Scaler Loaded Successfully!")
    print("=" * 50)

    return scaler

# Load New Transaction

def load_transaction(file_path):

    transaction = pd.read_csv(file_path)

    print("=" * 50)
    print("Transaction Loaded Successfully!")
    print(transaction.head())
    print("=" * 50)

    return transaction

# Preprocess Transaction

def preprocess_transaction(transaction, scaler):

    transaction = transaction.copy()

    transaction[["Time", "Amount"]] = scaler.transform(
        transaction[["Time", "Amount"]]
    )

    print("=" * 50)
    print("Transaction Preprocessed Successfully!")
    print("=" * 50)

    return transaction

# Predict Transaction

def predict_transaction(model, transaction):

    prediction = model.predict(transaction)[0]

    probability = model.predict_proba(transaction)[0][1]

    print("=" * 50)
    print("Prediction Result")
    print("=" * 50)

    if prediction == 1:

        print("Fraudulent Transaction Detected!")

    else:

        print("Normal Transaction")

    print(f"Fraud Probability: {probability:.4f}")

    print("=" * 50)

    return prediction, probability

# Main Function

def main():

    print("=" * 60)
    print("FraudShield AI - Prediction")
    print("=" * 60)

    # Load Model
    model = load_model()

    # Load Scaler
    scaler = load_scaler()

    # Load Transaction
    transaction = load_transaction(
        r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\data\new\transaction.csv"
    )

    # Preprocess Transaction
    transaction = preprocess_transaction(
        transaction,
        scaler
    )

    # Predict
    prediction, probability = predict_transaction(
        model,
        transaction
    )

    print("\n" + "=" * 60)
    print("Prediction Completed Successfully!")
    print("=" * 60)

# Run

if __name__ == "__main__":

    main()