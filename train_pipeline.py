import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
import os

# Create artifacts directory if it doesn't exist
os.makedirs("artifacts", exist_ok=True)

def train_model():
    print("Loading data...")
    # NOTE: Update this path to where your gemstone.csv actually resides
    # For now, I'm assuming it's in a data folder or same directory
    try:
        df = pd.read_csv('data/raw/gemstone.csv') 
    except FileNotFoundError:
        print("Error: 'gemstone.csv' not found. Please check the path in train_pipeline.py")
        return

    # 1. Data Cleaning
    if 'id' in df.columns:
        df = df.drop(labels=['id'], axis=1)

    X = df.drop(labels=['price'], axis=1)
    y = df[['price']]

    # 2. Define Features
    categorical_cols = X.select_dtypes(include='object').columns
    numerical_cols = X.select_dtypes(exclude='object').columns

    # 3. Define Ordinal Categories (Crucial from notebook)
    cut_categories = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
    clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

    # 4. Build Pipeline
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinalencoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories])),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num_pipeline', num_pipeline, numerical_cols),
        ('cat_pipeline', cat_pipeline, categorical_cols)
    ])

    # 5. Initialize Model (XGBoost)
    model = XGBRegressor(n_estimators=100, random_state=42)

    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # 6. Train and Save
    print("Training model...")
    final_pipeline.fit(X, y)

    save_path = os.path.join("artifacts", "model.pkl")
    joblib.dump(final_pipeline, save_path)
    print(f"Model saved successfully at {save_path}")

if __name__ == "__main__":
    train_model()