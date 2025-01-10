from phi.model.groq import Groq
from dotenv import load_dotenv
import pandas as pd
import pydicom  # For DICOM files
import os
from phi.agent import Agent
from io import BytesIO
import requests
load_dotenv()

def get_data(source):
    """Loads data from a given source (CSV or DICOM)."""
    try:
        if source.startswith("http"): #check if it is a URL
            if source.endswith(".csv"):
                response = requests.get(source)
                response.raise_for_status()  # Raise an exception for bad status codes
                df = pd.read_csv(BytesIO(response.content))
                return df
            elif source.endswith((".dcm", ".dicom")):
                response = requests.get(source, stream=True)
                response.raise_for_status()
                ds = pydicom.dcmread(BytesIO(response.content))
                return ds
            else:
                return f"Error: Unsupported file format from URL. Only CSV and DICOM files are supported. URL given: {source}"
        elif os.path.isfile(source): #check if it is a file path
            if source.endswith(".csv"):
                df = pd.read_csv(source)
                return df
            elif source.endswith((".dcm", ".dicom")):
                ds = pydicom.dcmread(source)
                return ds
            else:
                return f"Error: Unsupported file format from file. Only CSV and DICOM files are supported. File given: {source}"
        else:
            return f"Error: Invalid source provided. Please provide a valid URL or file path. Source given: {source}"

    except requests.exceptions.RequestException as e:
        return f"Error fetching data from URL: {e}"
    except pd.errors.ParserError as e:
        return f"Error parsing CSV: {e}"
    except pydicom.errors.InvalidDicomError as e:
        return f"Error reading DICOM file: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"



def analyze_data(question, source=None, data=None):
    """Analyzes data based on the user's question."""

    if data is None and source:
        data = get_data(source)
        if isinstance(data, str):
            return data  # Return the error message
    elif data is None and not source:
        return "Please provide a source (URL or file path) for the data."

    if isinstance(data, pd.DataFrame):
        answer = f"CSV Data was found and loaded. Here is the first 5 rows: \n{data.head().to_markdown(index=False)}"
        # Add more sophisticated analysis based on the question here
    elif isinstance(data, pydicom.dataset.FileDataset):
        answer = f"DICOM Data was found and loaded. Here is a summary:\n{data}"  # Display DICOM metadata
        # You can access specific DICOM tags like this:
        # patient_name = data.PatientName
        # ... and include them in the answer.
    else:
        answer = "No data was found or data format not recognized. Please check the source."
    return answer

head_agent = Agent(
    name="Healthcare Data Enrichment Agent",
    model=Groq(id="llama-3.3-70b-versatile"),  # Replace with your preferred model
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

head_agent.get_data = get_data
head_agent.analyze_data = analyze_data

# Example usage (CSV from URL):
csv_url = "https://raw.githubusercontent.com/plotly/datasets/master/titanic.csv"  # Example CSV URL
user_question_csv = "Summarize the CSV dataset."
answer_csv = head_agent.run(head_agent.analyze_data, question=user_question_csv, source=csv_url)
print(answer_csv)

# Example usage (DICOM from URL - you'll need a public DICOM URL)
dicom_url = "https://www.kaggle.com/datasets/kmader/siim-medical-images?select=dicom_dir"
user_question_dicom = "Summarize the DICOM data."
answer_dicom = head_agent.run(head_agent.analyze_data, question=user_question_dicom, source=dicom_url)
print(answer_dicom)

# Example usage (local DICOM file):
# local_dicom_path = "path/to/your/local/file.dcm" # Replace with your local path
# user_question_local_dicom = "Summarize the local DICOM data."
# answer_local_dicom = head_agent.run(head_agent.analyze_data, question=user_question_local_dicom, source=local_dicom_path)
# print(answer_local_dicom)