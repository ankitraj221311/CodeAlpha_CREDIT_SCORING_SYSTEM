import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(df):

    encoder = LabelEncoder()

    for col in df.select_dtypes(include='object'):
        df[col] = encoder.fit_transform(df[col])

    return df