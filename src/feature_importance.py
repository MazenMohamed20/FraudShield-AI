import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_model():
    model_path= r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\models\random_forest.pkl"
    model = joblib.load(model_path)
    print("=" * 50)
    print("Model Loaded Successfully!")
    print("=" * 50)
    
    return model


# Load Processed Training Data

def load_training_data():

    processed_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\data\processed"

    X_train = pd.read_csv(
        os.path.join(processed_path, "X_train.csv")
    )

    print("=" * 50)
    print("Training Data Loaded Successfully!")
    print(f"Shape: {X_train.shape}")
    print("=" * 50)

    return X_train

# Extract Feature Importance

def get_feature_importance(model, X_train):

    importance = pd.DataFrame({

        "Feature": X_train.columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    ).reset_index(drop=True)

    print("=" * 50)
    print("Top 15 Important Features")
    print("=" * 50)

    print(importance.head(15))

    return importance

# Plot Feature Importance

def plot_feature_importance(importance):

    reports_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\reports\figures"

    os.makedirs(reports_path, exist_ok=True)

    plt.figure(figsize=(10, 8))

    sns.barplot(
        data=importance.head(15),
        x="Importance",
        y="Feature"
    )

    plt.title("Top 15 Feature Importance - Random Forest")

    plt.xlabel("Importance Score")

    plt.ylabel("Features")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            reports_path,
            "random_forest_feature_importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print("=" * 50)
    print("Feature Importance Plot Saved Successfully!")
    print("=" * 50)

# Save Feature Importance

def save_feature_importance(importance):

    reports_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\reports"

    os.makedirs(reports_path, exist_ok=True)

    importance.to_csv(

        os.path.join(
            reports_path,
            "feature_importance.csv"
        ),

        index=False

    )

    print("=" * 50)
    print("Feature Importance Saved Successfully!")
    print("=" * 50)

# Main Function

def main():

    print("=" * 60)
    print("FraudShield AI - Feature Importance")
    print("=" * 60)

    # Load Model
    model = load_model()

    # Load Training Data
    X_train = load_training_data()

    # Extract Feature Importance
    importance = get_feature_importance(
        model,
        X_train
    )

    # Plot Feature Importance
    plot_feature_importance(
        importance
    )

    # Save Feature Importance
    save_feature_importance(
        importance
    )

    print("\n" + "=" * 60)
    print("Feature Importance Analysis Completed Successfully!")
    print("=" * 60)

# Run
if __name__ == "__main__":

    main()