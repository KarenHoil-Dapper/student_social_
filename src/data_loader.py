import pandas as pd

def load_data(filepath):
    """
    Carga los datos desde un archivo Excel y retorna un DataFrame.
    """
    return pd.read_excel(filepath)