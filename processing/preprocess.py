import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from transform import AlpacaProcessor




if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    dp = AlpacaProcessor()

    df = pd.read_parquet(f"{base_dir}/input/stock_dataset.parquet")

    df = dp.clean_data(df)
    df = dp.add_technical_indicators(df)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    numerical_cols = df.select_dtypes(
            include=['float64', 'int64']).columns

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

    train.to_parquet(
        f"{base_dir}/train/train.parquet", index=False
    )
    validation.to_parquet(
        f"{base_dir}/validation/validation.parquet", index=False
    )
    test.to_parquet(f"{base_dir}/test/test.parquet", index=False)