import kaggle
import os
import pandas as pd
import pydicom
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set your Kaggle credentials (if not already set in your environment)
os.environ['KAGGLE_USERNAME'] = "mohananagasaimuvva" 
os.environ['KAGGLE_KEY'] = "a2edee09cefd840f54f8db07c5382fe8" 

DATASET_NAME = "kmader/siim-medical-images"

# Data Integration Agent: Collects data from Kaggle
def get_data(file_name):
    """Loads data from a Kaggle dataset without downloading the entire dataset."""
    try:
        # Get the Kaggle API client
        api = kaggle.KaggleApi()
        api.authenticate()

        # Download the specific file to a temporary location
        download_path = api.dataset_download_file(DATASET_NAME, file_name, path="./", force=True, quiet=True)

        if file_name.endswith(".csv"):
            df = pd.read_csv(download_path)
            os.remove(download_path)  # Remove the file after reading
            return df
        elif file_name.endswith((".dcm", ".dicom")):
            ds = pydicom.dcmread(download_path)
            os.remove(download_path)  # Remove the file after reading
            return ds
        else:
            os.remove(download_path)  # Remove the file even if it's not supported
            return f"Error: Unsupported file format. Only CSV and DICOM files are supported. File given: {file_name}"

    except Exception as e:
        return f"Error during data retrieval: {e}"


# Diagnostic Support Agent: Analyzes data to assist clinicians
def analyze_data(file_name, question):
    """Analyzes data based on the user's question."""
    try:
        data = get_data(file_name)
        if isinstance(data, str):
            return f"### Error\n{data}"

        if isinstance(data, pd.DataFrame):
            summary = f"""### Data Analysis Summary
#### Dataset Overview
- Number of rows: {len(data)}
- Number of columns: {len(data.columns)}
- Column names: {', '.join(data.columns)}

#### Sample Data
{data.head().to_markdown()}

#### Analysis
{question}
"""
            return summary
            
        elif isinstance(data, pydicom.dataset.FileDataset):
            summary = f"""### DICOM Analysis Summary
#### File Information
- Patient ID: {data.PatientID if 'PatientID' in data else 'N/A'}
- Modality: {data.Modality if 'Modality' in data else 'N/A'}
- Study Date: {data.StudyDate if 'StudyDate' in data else 'N/A'}

#### Analysis
{question}
"""
            return summary
            
        return "### Error\nUnsupported data format"

    except Exception as e:
        return f"### Error\nError analyzing data: {e}"


# Head Agent: Combines Data Integration and Diagnostic Support
head_agent = Agent(
    name="Healthcare Data Enrichment Agent",
    model=OpenAIChat(model="gpt-4"),
    instructions="""
    You are a healthcare data enrichment assistant. When analyzing data:
    - Collect relevant data from Kaggle using the Data Integration Agent.
    - Perform detailed analysis using the Diagnostic Support Agent.
    - Provide clear summaries of the data and analyses in markdown format.
    """,
    show_tool_calls=True,
    markdown=True
)

# Attach tools (sub-agents) to the Head Agent
head_agent.tools = {
    "data_integration": get_data,
    "diagnostic_support": analyze_data
}

# Function to run analysis via the Head Agent
def run_analysis(question, file_name):
    """Handles the analysis flow using the Head Agent."""
    try:
        # Use Diagnostic Support Agent for analysis
        result = head_agent.tools["diagnostic_support"](file_name=file_name, question=question)
        return result
        
    except Exception as e:
        return f"Error during analysis: {str(e)}"

# Example usage
if __name__ == "__main__":
    # CSV example
    file_name_csv = "metadata.csv"
    user_question_csv = "Summarize the metadata."
    print("\nCSV Analysis:")
    print(run_analysis(user_question_csv, file_name_csv))

    # DICOM example
    file_name_dicom = "ID_0007_AGE_0061_CONTRAST_1_CT.dcm"
    user_question_dicom = "Summarize the DICOM data."
    print("\nDICOM Analysis:")
    print(run_analysis(user_question_dicom, file_name_dicom))
