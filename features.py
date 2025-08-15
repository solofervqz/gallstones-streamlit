# features.py
import pandas as pd
import numpy as np

COLUMNS_WITH_COMMAS = [
    'Weight', 'Body Mass Index (BMI)', 'Total Body Water (TBW)',
    'Extracellular Water (ECW)', 'Intracellular Water (ICW)',
    'Extracellular Fluid/Total Body Water (ECF/TBW)',
    'Total Body Fat Ratio (TBFR) (%)', 'Lean Mass (LM) (%)',
    'Body Protein Content (Protein) (%)', 'Visceral Fat Rating (VFR)',
    'Bone Mass (BM)', 'Muscle Mass (MM)', 'Obesity (%)',
    'Total Fat Content (TFC)', 'Visceral Fat Area (VFA)',
    'Visceral Muscle Area (VMA) (Kg)', 'Hepatic Fat Accumulation (HFA)',
    'Glucose', 'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)',
    'High Density Lipoprotein (HDL)', 'Triglyceride',
    'Aspartat Aminotransferaz (AST)', 'Alanin Aminotransferaz (ALT)',
    'Alkaline Phosphatase (ALP)', 'Creatinine',
    'Glomerular Filtration Rate (GFR)', 'C-Reactive Protein (CRP)',
    'Hemoglobin (HGB)', 'Vitamin D'
]

TARGET_COL = 'Gallstone Status'

def convert_commas_to_dots(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in COLUMNS_WITH_COMMAS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def make_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe DF crudo con columnas originales.
    Devuelve DF transformado con las nuevas features y dummies necesarias.
    """
    df = df_raw.copy()

    # Ratios bioquímicos (evitar div/0)
    eps = 1e-5
    if {'Alanin Aminotransferaz (ALT)', 'Aspartat Aminotransferaz (AST)'}.issubset(df.columns):
        df['ALT_AST_ratio'] = df['Alanin Aminotransferaz (ALT)'] / (df['Aspartat Aminotransferaz (AST)'] + eps)
    if {'Total Cholesterol (TC)', 'High Density Lipoprotein (HDL)'}.issubset(df.columns):
        df['Chol_HDL_ratio'] = df['Total Cholesterol (TC)'] / (df['High Density Lipoprotein (HDL)'] + eps)
    if {'Low Density Lipoprotein (LDL)', 'High Density Lipoprotein (HDL)'}.issubset(df.columns):
        df['LDL_HDL_ratio'] = df['Low Density Lipoprotein (LDL)'] / (df['High Density Lipoprotein (HDL)'] + eps)
    if {'Triglyceride', 'High Density Lipoprotein (HDL)'}.issubset(df.columns):
        df['TG_HDL_ratio'] = df['Triglyceride'] / (df['High Density Lipoprotein (HDL)'] + eps)

    # Grupos de edad (categoría → dummies)
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 70, 120], labels=['<30', '30-50', '50-70', '70+'])
        df = pd.get_dummies(df, columns=['Age_Group'], drop_first=True)

    # Flags binarios clínicos
    if 'Body Mass Index (BMI)' in df.columns:
        df['Obese'] = (df['Body Mass Index (BMI)'] >= 30).astype(int)
    if 'Total Cholesterol (TC)' in df.columns:
        df['High_Cholesterol'] = (df['Total Cholesterol (TC)'] > 200).astype(int)
    if 'Glucose' in df.columns:
        df['Hyperglycemia'] = (df['Glucose'] > 110).astype(int)
    if 'Vitamin D' in df.columns:
        df['Low_VitD'] = (df['Vitamin D'] < 20).astype(int)

    # Interacciones simples
    if {'Body Mass Index (BMI)', 'Age'}.issubset(df.columns):
        df['BMI_Age'] = df['Body Mass Index (BMI)'] * df['Age']
    if {'Visceral Fat Area (VFA)', 'Alanin Aminotransferaz (ALT)'}.issubset(df.columns):
        df['VFA_ALT'] = df['Visceral Fat Area (VFA)'] * df['Alanin Aminotransferaz (ALT)']

    return df

def split_X_y(df: pd.DataFrame):
    y = df[TARGET_COL].copy() if TARGET_COL in df.columns else None
    X = df.drop(columns=[TARGET_COL]) if TARGET_COL in df.columns else df.copy()
    return X, y
