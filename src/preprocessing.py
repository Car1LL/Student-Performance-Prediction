import pandas as pd
from typing import Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

def create_class_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new categorical column in the original DataFrame.

    Args:
        df: Original pandas DataFrame

    Returns:
        Returns mutated DataFrame
    """
    
    df['G3_class'] = pd.cut(df['G3'], bins=3, labels=['C', 'B', 'A'])

    return df
    
def delete_continuous_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes continuous target columns from the dataset to prevent data leakage.

    Args:
        df: Original pandas DataFrame

    Returns:
        Returns mutated DataFrame
    """
    
    return df.drop(columns=['G1', 'G2', 'G3'])

def split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits original DataFrame into X and y subsets. Maps target variable into numerical values.

    Args:
        df: Original pandas DataFrame

    Returns:
        Tuple of X and y values as pandas DataFrame.
    """
    X = df.drop(columns=['G3_class'])
    y = df['G3_class'].map({
        "A": 2,
        "B": 1,
        "C": 0
    })

    return X, y

def create_preprocessor(
        categorical_features: pd.Index,
        numerical_features: pd.Index,
        scale_numeric: bool = True,
        encode_categorical: str = "onehot"
) -> ColumnTransformer:
    """
    Creates preprocessor for the dataset, handling both: numerical and categorical features.

    Args:
        categorical_features: a pandas Index object, storing categorical columns
        numerical_features: a pandas Index object, storing numerical columns
        scale_numeric: Use StandardScaler if True, otherwise ignore any numerical scaler
        encode_categorical: Option for encoding Categorical features

    Returns:
        preprocessor for the data.
    """

    num_transformer = (StandardScaler() if scale_numeric else "passthrough")

    match encode_categorical:
        case "onehot":
            cat_transformer = OneHotEncoder(handle_unknown="ignore")
        case "passthrough":
            cat_transformer = "passthrough"
        
        case _:
            raise ValueError(f"Unknown encoding strategy: {encode_categorical}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, categorical_features),
            ("num", num_transformer, numerical_features)
        ]
    )

    return preprocessor


