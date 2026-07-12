import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE


# Load Dataset
def load_data(path):

    data = pd.read_csv(path)

    print("=" * 50)
    print("Dataset Loaded Successfully!")
    print(f"Shape: {data.shape}")
    print("=" * 50)

    return data


# Remove Duplicate Rows
def remove_duplicates(data):

    duplicates = data.duplicated().sum()

    print("=" * 50)
    print("Checking Duplicate Rows...")
    print(f"Duplicate Rows: {duplicates}")

    if duplicates > 0:
        data = data.drop_duplicates().reset_index(drop=True)
        print("Duplicates Removed Successfully!")
    else:
        print("No Duplicate Rows Found.")

    print("=" * 50)

    return data


# Split Features and Target
def split_features_target(data):

    X = data.drop("Class", axis=1)

    y = data["Class"]

    print("=" * 50)
    print("Features and Target Split Completed!")
    print(f"Features Shape : {X.shape}")
    print(f"Target Shape   : {y.shape}")
    print("=" * 50)

    return X, y


# Feature Scaling
def feature_scaling(X_train, X_test):

    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = RobustScaler()

    X_train[["Time", "Amount"]] = scaler.fit_transform(
        X_train[["Time", "Amount"]]
    )

    X_test[["Time", "Amount"]] = scaler.transform(
        X_test[["Time", "Amount"]]
    )

    print("=" * 50)
    print("Feature Scaling Completed!")
    print("=" * 50)

    return X_train, X_test, scaler


# Train Test Split
def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y

    )

    print("=" * 50)
    print("Train/Test Split Completed!")
    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"y_test  : {y_test.shape}")
    print("=" * 50)

    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):

    smote = SMOTE(random_state=42)

    X_train, y_train = smote.fit_resample(
        X_train,
        y_train
    )

    print("=" * 50)
    print("SMOTE Applied Successfully!")
    print(y_train.value_counts())
    print("=" * 50)

    return X_train, y_train

# Save Processed Data
def save_processed_data(

    X_train,
    X_test,
    y_train,
    y_test

):

    processed_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\data\processed"

    os.makedirs(processed_path, exist_ok=True)

    X_train.to_csv(
        os.path.join(processed_path, "X_train.csv"),
        index=False
    )

    X_test.to_csv(
        os.path.join(processed_path, "X_test.csv"),
        index=False
    )

    y_train.to_frame().to_csv(
        os.path.join(processed_path, "y_train.csv"),
        index=False
    )

    y_test.to_frame().to_csv(
        os.path.join(processed_path, "y_test.csv"),
        index=False
    )

    print("=" * 50)
    print("Processed Data Saved Successfully!")
    print("=" * 50)


# Save Scaler
def save_scaler(scaler):

    models_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\models"

    os.makedirs(models_path, exist_ok=True)

    joblib.dump(
        scaler,
        os.path.join(models_path, "robust_scaler.pkl")
    )

    print("=" * 50)
    print("Scaler Saved Successfully!")
    print("=" * 50)


# Verify Saved Files
def verify_files():

    files = [

        r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\data\processed\X_train.csv",

        r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\data\processed\X_test.csv",

        r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\data\processed\y_train.csv",

        r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\data\processed\y_test.csv",

        r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\models\robust_scaler.pkl"

    ]

    print("=" * 50)
    print("Verification")

    for file in files:

        if os.path.exists(file):
            print(f"✔ {file}")
        else:
            print(f"✘ {file}")

    print("=" * 50)

# Main
def main():

    data = load_data(
        r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\data\raw\creditcard.csv"
    )

    data = remove_duplicates(data)

    X, y = split_features_target(data)

    X_train, X_test, y_train, y_test = split_data(
              X,
              y
    )

    X_train, y_train = apply_smote(
            X_train,
            y_train
    )

    X_train, X_test, scaler = feature_scaling(
              X_train,
              X_test
    )

    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test
    )

    save_scaler(
        scaler
    )

    verify_files()


# Run
if __name__ == "__main__":

    main()