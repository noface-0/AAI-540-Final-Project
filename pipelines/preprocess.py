import os
import copy
import argparse
import pandas as pd
import numpy as np
import boto3
import logging
import pathlib
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def process_data(df):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols)
        ]
    )

    df_transformed = preprocess.fit_transform(df)

    train, validation, test = np.split(
        df_transformed.sample(frac=1), 
        [int(0.7 * len(df)), int(0.85 * len(df))]
    )

    train.to_parquet(f"{BASE_DIR}/train/train.parquet", index=False)
    validation.to_parquet(f"{BASE_DIR}/validation/validation.parquet", index=False)
    test.to_parquet(f"{BASE_DIR}/test/test.parquet", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input data for training.")
    parser.add_argument("--input-data", type=str, required=True,
                        help="The input data.")
    args = parser.parse_args()

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    input_data = args.input_data
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])

    logging.info("Downloading data from bucket: %s, key: %s", bucket, key)
    fn = f"{base_dir}/data/extracted_stocks.parquet"
    s3 = boto3.resource("s3")
    s3.Bucket(bucket).download_file(key, fn)

    input_df = pd.read_parquet(fn)

    process_data(input_df)