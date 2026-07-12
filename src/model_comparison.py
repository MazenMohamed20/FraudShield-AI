import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create Model Comparison Table

def create_comparison_table():

    results = {

        "Model": [

            "Logistic Regression",

            "Decision Tree",

            "Random Forest",

            "Neural Network"

        ],

        "Accuracy": [

            0.9915,

            0.9905,

            0.9995,

            0.9993

        ],

        "Precision": [

            0.1481,

            0.1273,

            0.8902,

            0.7957

        ],

        "Recall": [

            0.8526,

            0.8000,

            0.7684,

            0.7789

        ],

        "F1 Score": [

            0.2523,

            0.2197,

            0.8249,

            0.7872

        ],

        "ROC AUC": [

            0.9626,

            0.8808,

            0.9739,

            0.9432

        ]

    }

    comparison = pd.DataFrame(results)

    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)

    print(comparison)

    return comparison

# Save Comparison Table

def save_comparison_table(comparison):

    reports_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\reports"

    os.makedirs(reports_path, exist_ok=True)

    comparison.to_csv(

        os.path.join(
            reports_path,
            "model_comparison.csv"
        ),

        index=False

    )

    print("=" * 50)
    print("Model Comparison Saved Successfully!")
    print("=" * 50)

# Plot Model Comparison

def plot_model_comparison(comparison):

    reports_path = r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\reports\figures"

    os.makedirs(reports_path, exist_ok=True)

    metrics = [

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score",

        "ROC AUC"

    ]

    for metric in metrics:

        plt.figure(figsize=(8,5))

        sns.barplot(

            data=comparison,

            x="Model",

            y=metric

        )

        plt.title(f"{metric} Comparison")

        plt.xticks(rotation=15)

        plt.tight_layout()

        plt.savefig(

            os.path.join(

                reports_path,

                f"{metric.lower().replace(' ','_')}_comparison.png"

            ),

            dpi=300,

            bbox_inches="tight"

        )

        plt.show()

    print("=" * 50)
    print("Comparison Plots Saved Successfully!")
    print("=" * 50)

# Get Best Model

def get_best_model(comparison):

    best_model = comparison.loc[
        comparison["F1 Score"].idxmax()
    ]

    print("=" * 60)
    print("Best Model")
    print("=" * 60)

    print(f"Model      : {best_model['Model']}")
    print(f"Accuracy   : {best_model['Accuracy']:.4f}")
    print(f"Precision  : {best_model['Precision']:.4f}")
    print(f"Recall     : {best_model['Recall']:.4f}")
    print(f"F1 Score   : {best_model['F1 Score']:.4f}")
    print(f"ROC AUC    : {best_model['ROC AUC']:.4f}")

    print("=" * 60)

    return best_model

# Main Function

def main():

    print("=" * 60)
    print("FraudShield AI - Model Comparison")
    print("=" * 60)

    # Create Comparison Table
    comparison = create_comparison_table()

    # Save Comparison
    save_comparison_table(comparison)

    # Plot Comparison
    plot_model_comparison(comparison)

    # Best Model
    get_best_model(comparison)

    print("\n" + "=" * 60)
    print("Model Comparison Completed Successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
