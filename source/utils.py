import joblib
import pandas as pd

def import_data():
    """
    Import raw data from a CSV file.

    Args:
    - raw_data: File path to the raw data CSV file.

    Returns:
    - DataFrame: Pandas DataFrame containing the raw data.
    """
    # Load raw data from CSV file
    raw_data = pd.read_csv('./files/raw_BankChurners.csv')
    return raw_data

def save_model(model):
    """
    Save a trained model to a pickle file.

    Args:
    - model: Trained model to be saved.
    - file_name: Name of the file for the pickle file to be saved.
    """
    # Define file path for saving the model
    file_path = f'./files/{model}.pkl'
    # Save the trained model to a pickle file
    joblib.dump(model, file_path)