import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class preprocessing:
    
    @staticmethod
    def data_loading(data_path: str):
        """
        Load CSV data into DataFrame
        """
        df = pd.read_csv(data_path)
        return df
    
    @staticmethod
    def create_rul(df):
        """
        Create RUL (Remaining Useful Life) column
        """
        max_wear = df["Tool wear [min]"].max()
        df["RUL"] = max_wear - df["Tool wear [min]"]
        return df

    @staticmethod
    def split_data(df):
        """
        Split data into train, validation, and test sets
        """
        train, test = train_test_split(df, test_size=0.15, random_state=42)
        train, val = train_test_split(train, test_size=0.15, random_state=42)
        return train, val, test

    @staticmethod
    def build_preprocessor(df):
        """
        Build a preprocessor for numeric and categorical columns
        """
        numeric = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical = df.select_dtypes(include=['object']).columns.tolist()

        # Remove target columns if present
        numeric = [c for c in numeric if c not in ["Machine failure", "RUL"]]

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        ])

        return preprocessor, numeric, categorical