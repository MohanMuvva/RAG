import pandas as pd

def get_patient_medication(file_path, patient_name):
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Normalize the 'Name' column
    data['Name'] = data['Name'].str.lower().str.strip()
    
    # Normalize the input patient name
    patient_name = patient_name.lower().strip()
    
    # Filter the DataFrame for the specified patient
    patient_data = data[data['Name'] == patient_name]
    
    # Check if patient data is found
    if not patient_data.empty:
        # Extract the medication
        medication = patient_data['Medication'].values[0]
        return medication
    else:
        return "No medication information found for the specified patient."

if __name__ == "__main__":
    file_path = "cleaned_healthcare_dataset.csv"
    patient_name = "Jasmine Aguilar"
    medication = get_patient_medication(file_path, patient_name)
    print(medication)