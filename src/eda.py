import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
def load_data(path):
    data = pd.read_csv(path)
    print("Dataset Loaded Successfully!")
    return data

# Class Distribution
def class_distribution(data):

    class_counts = data["Class"].value_counts()

    plt.figure(figsize=(6,5))

    plt.bar(
        class_counts.index.astype(str),
        class_counts.values,
        color=["steelblue","red"]
    )

    plt.title("Distribution of Classes")
    plt.xlabel("Class")
    plt.ylabel("Count")

    plt.xticks([0,1],["Normal","Fraud"])

    for i, value in enumerate(class_counts.values):
        plt.text(
            i,
            value,
            f"{value:,}",
            ha="center",
            va="bottom"
        )

    plt.grid(axis="y")

    plt.show()

# Amount Distribution
def amount_distribution(data):

    plt.figure(figsize=(8,5))

    plt.hist(
        data["Amount"],
        bins=50,
        edgecolor="black",
        color="green"
    )

    plt.title("Distribution of Amount")
    plt.xlabel("Amount")
    plt.ylabel("Count")

    plt.grid(True)

    plt.show()

# Time Distribution
def time_distribution(data):

    plt.figure(figsize=(8,5))

    plt.hist(
        data["Time"],
        bins=50,
        edgecolor="black",
        color="orange"
    )

    plt.title("Distribution of Time")
    plt.xlabel("Time")
    plt.ylabel("Count")

    plt.grid(True)

    plt.show()

# Amount Box Plot
def amount_boxplot(data):

    plt.figure(figsize=(6,6))

    plt.boxplot(data["Amount"])

    plt.title("Box Plot of Transaction Amount")
    plt.ylabel("Amount")

    plt.grid(True)

    plt.show()

# Pie Chart
def class_percentage(data):

    counts = data["Class"].value_counts()

    plt.figure(figsize=(6,6))

    plt.pie(
        counts,
        labels=["Normal","Fraud"],
        autopct="%1.2f%%",
        startangle=90
    )

    plt.title("Fraud vs Normal Transactions")

    plt.show()

# Correlation Matrix
def correlation_matrix(data):

    plt.figure(figsize=(18,12))

    correlation = data.corr()

    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0
    )

    plt.title("Correlation Matrix")

    plt.show()

# Duplicate Check
def check_duplicates(data):

    duplicates = data.duplicated().sum()

    print("="*50)
    print(f"Duplicate Rows : {duplicates}")
    print("="*50)

# Dataset Summary
def dataset_summary(data):

    print("="*50)
    print("Dataset Shape")
    print(data.shape)

    print("="*50)
    print("Missing Values")
    print(data.isnull().sum())

    print("="*50)
    print("Data Types")
    print(data.dtypes)

# Main Function
def main():

    data = load_data(r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\data\raw\creditcard.csv")

    dataset_summary(data)

    class_distribution(data)

    amount_distribution(data)

    time_distribution(data)

    amount_boxplot(data)

    class_percentage(data)

    correlation_matrix(data)

    check_duplicates(data)


if __name__ == "__main__":
    main()