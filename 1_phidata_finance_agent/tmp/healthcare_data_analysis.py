import pandas as pd

def load_healthcare_data(file_path):
    # Load the healthcare dataset
    df = pd.read_csv(file_path)
    return df

def get_columns_and_types(df):
    # Get the column names and their data types
    return df.dtypes

if __name__ == "__main__":
    file_path = "C:\\Users\\saich\\ai-agents\\1_phidata_finance_agent\\cleaned_healthcare_dataset.csv"
    df = load_healthcare_data(file_path)
    column_types = get_columns_and_types(df)
    print(column_types)