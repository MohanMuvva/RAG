import pandas as pd

def get_medication_for_patient(patient_name):
    # Reading the dataset
    df = pd.read_csv('cleaned_healthcare_dataset.csv')
    
    # Converting the Name column to lowercase for case-insensitive comparison
    df['Name'] = df['Name'].str.lower()
    
    # Searching for the patient by name
    medication = df[df['Name'] == patient_name.lower()]['Medication'].values
    
    # Check if medication is found
    if medication:
        return medication[0]
    else:
        return 'Patient not found.'

# Name of the patient
target_patient_name = 'JASmINe aGuIlaR'

# Running the function to get medication
medication_for_patient = get_medication_for_patient(target_patient_name)