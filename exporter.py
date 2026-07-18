import pandas as pd

def save_results(df, path):
    df.to_excel(path, index=False)

def save_csv(df, path):
    df.to_csv(path, index=False)