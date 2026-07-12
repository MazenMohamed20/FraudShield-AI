import pandas as pd

#Loading the dataset
data=pd.read_csv(r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\data\raw\creditcard.csv")
print("Data loaded successfully")

# Displaying the first 5 rows of the dataset
print(f"First 5 rows of the dataset:\n{data.head()}") #displays the first 5 rows of the dataset with columns v1 - v28, Time, Amount, Class

# Displaying the shape of the dataset
print(f"Shape of the dataset: {data.shape}") #(284807, 31)

# Displaying the column names of the dataset
print(f"Column names of the dataset: {data.columns.tolist()}") # v1 - v28, Time, Amount, Class

# Displaying the data types of each column
print(f"Data types of each column:\n{data.dtypes}") #all columns are of type float64 except for the 'Class' column which is of type int64

# Displaying the number of missing values in each column
print(f"Number of missing values in each column:\n{data.isnull().sum()}") # none of the columns have missing values
      
# Displaying the summary statistics of the dataset
print(f"Summary statistics of the dataset:\n{data.describe()}") # for numerical columns, it provides count, mean, std, min, 25%, 50%, 75%, max

# Displaying the number of unique values in each column
print(f"Number of unique values in each column:\n{data.nunique()}") #none of the columns have unique values except for the target variable 'Class' which has 2 unique values (0: non-fraudulent, 1: fraudulent)

# Displaying the distribution of the target variable
print(f"Distribution of the target variable:\n{data['Class'].value_counts()}") # displays the count of each class (0: non-fraudulent, 1: fraudulent)


"""
data=data.drop("Class",axis=1) #dropping the target variable 'Class' from the dataset for further analysis
data.to_csv(r"E:\Internships\FraudShield AI - Intelligent Credit Card Fraud Detection Platform\data\test\transactions.csv",index=False) #saving the processed dataset to a new csv file

print("Processed dataset saved successfully to transactions.csv")
"""
