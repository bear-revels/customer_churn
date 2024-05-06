import joblib
import pandas as pd

import pandas as pd

def import_data(file_path):
    """
    Import raw data from a CSV file.

    Args:
    - file_path: File path to the raw data CSV file.

    Returns:
    - df: Pandas DataFrame containing the raw data.
    - numeric_columns: List of numeric columns in the DataFrame.
    - categorical_columns: List of categorical columns in the DataFrame.
    """
    # Load raw data from CSV file
    df = pd.read_csv(file_path)

    # Extract numeric and categorical columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    return df, numeric_columns, categorical_columns

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

