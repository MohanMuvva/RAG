import pandas as pd

def load_and_summarize_dataset(file_path):
    try:
        # Load the dataset
        data = pd.read_csv(file_path)
        
        # Get basic information and headers
        headers = data.columns.tolist()
        summary = data.describe(include='all').to_string()
        
        print("Data loaded successfully!")
        print("\nHeaders: ", headers)
        print("\nSummary: ", summary)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    file_path = "C:\\Users\\saich\\ai-agents\\1_phidata_finance_agent\\cleaned_healthcare_dataset.csv"
    load_and_summarize_dataset(file_path)