import pandas as pd

def get_medication(patient_name):
    # Load dataset with the correct path
    df = pd.read_csv('C:/Users/saich/Favorites/Mohan/GenerativeAI/Python/PythonPrograms/Data_Manpulation/cleaned_healthcare_dataset.csv')
    
    # Standardize the patient name for comparison
    df['Name'] = df['Name'].str.lower().str.strip()
    patient_name = patient_name.lower().strip()
    
    # Find the medication
    medication = df.loc[df['Name'] == patient_name, 'Medication']
    
    if not medication.empty:
        return medication.iloc[0]
    else:
        return 'Patient not found'

# Define the patient's name
patient_name = "JASmINe aGuIlaR"

# Get the medication for the patient
medication_for_patient = get_medication(patient_name)