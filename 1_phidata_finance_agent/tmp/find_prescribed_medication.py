import pandas as pd

def find_medication(patient_name):
    # Load the dataset
    df = pd.read_csv('cleaned_healthcare_dataset.csv')
    
    # Normalize the patient name to matching case
    patient_name_normalized = patient_name.strip().lower()
    
    # Find the medication for the specified patient
    patient_info = df[df['Name'].str.lower() == patient_name_normalized]
    
    if not patient_info.empty:
        return patient_info.iloc[0]['Medication']
    else:
        return "Patient not found"

if __name__ == "__main__":
    patient_name = "JASmINe aGuIlaR"
    medication = find_medication(patient_name)
    print(medication)