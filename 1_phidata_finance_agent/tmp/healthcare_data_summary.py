import pandas as pd

def summarize_healthcare_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Extract column names and data types
    column_info = df.dtypes
    
    # Provide descriptive statistics for numerical columns
    description = df.describe()
    
    # Number of entries
    num_entries = len(df)
    
    return column_info, description, num_entries

if __name__ == "__main__":
    # Path to the CSV file
    file_path = "C:\\Users\\saich\\ai-agents\\1_phidata_finance_agent\\cleaned_healthcare_dataset.csv"
    
    # Get the summary of the healthcare data
    column_info, description, num_entries = summarize_healthcare_data(file_path)
    
    print('Column Information:')
    print(column_info)
    print('\nDescriptive Statistics:')
    print(description)
    print(f'\nNumber of Entries: {num_entries}')