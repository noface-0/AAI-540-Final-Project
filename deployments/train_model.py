import os
import io
import argparse
import pandas as pd
import boto3
import logging
import torch

from environments.base import StockTradingEnv
from training.train_test import train, test
from config.indicators import INDICATORS
from config.tickers import DOW_30_TICKER
from config.models import ERL_PARAMS, SAC_PARAMS
from config.training import (
    TIME_INTERVAL,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    AGENT
)
from utils.utils import get_var
from deployments.s3_utils import (
    load_data_from_s3,
    load_model_from_local_path,
    save_model_to_s3,
    get_secret
)


def train_model(
        train_data=None, 
        validation_data=None,
        api_key=None,
        api_secret=None,
        api_url=None,
):
    # Initialize environment
    env = StockTradingEnv

    agent_configs = {
        "ppo": ERL_PARAMS,
        "sac": ERL_PARAMS
    }
    params = agent_configs.get(AGENT)

    if train_data.empty or validation_data.empty:
        logging.info("No training data provided. Auto-downloading.")
        split = True
    else:
        split = False

    # Training phase
    print("Starting training phase...")
    train(
        data=train_data,
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        time_interval=TIME_INTERVAL, 
        technical_indicator_list=INDICATORS,
        drl_lib='elegantrl',
        env=env,
        model_name=AGENT,
        if_vix=True,
        erl_params=params,
        cwd='models/runs/papertrading_erl',
        break_step=1e6,
        split=split,
        api_key=api_key,
        api_secret=api_secret,
        api_url=api_url
    )
    
    # Testing phase
    print("Starting testing phase...")
    account_value_erl = test(
        data=validation_data,
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        time_interval=TIME_INTERVAL,
        technical_indicator_list=INDICATORS,
        drl_lib='elegantrl',
        env=env,
        model_name=AGENT,
        if_vix=True,
        cwd='models/runs/papertrading_erl',
        net_dimension=params['net_dimension'],
        split=split,
        api_key=api_key,
        api_secret=api_secret,
        api_url=api_url
    )
    print(
        "Testing phase completed. Final account value:", 
        account_value_erl
    )

    full_data_df = pd.concat(
        [train_data, validation_data], ignore_index=True
    ) if train_data is not None else None

    print("Starting full data training phase...")
    train(
        data=full_data_df,
        start_date=TRAIN_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        time_interval=TIME_INTERVAL, 
        technical_indicator_list=INDICATORS,
        drl_lib='elegantrl',
        env=env,
        model_name=AGENT,
        if_vix=True,
        erl_params=params,
        cwd='models/runs/papertrading_erl_retrain',
        break_step=1e6,
        split=False,
        api_key=api_key,
        api_secret=api_secret,
        api_url=api_url
    )


if __name__ == "__main__":
    import os
    parser = argparse.ArgumentParser(description="Process input data for training.")
    parser.add_argument('--training', type=str, default=os.environ.get('S3_TRAINING'))
    parser.add_argument('--validation', type=str, default=os.environ.get('S3_VALIDATION'))
    args = parser.parse_args()

    logging.log(logging.DEBUG, args)
    logging.log(logging.DEBUG, os.environ)
    logging.log(logging.DEBUG, os.environ.get('S3_TRAINING'))
    logging.log(logging.DEBUG, os.environ.get('S3_VALIDATION'))
    # this should dynamically be set but something is wrong with env variables
    train_input = "s3://sagemaker-us-east-1-914326228175/DLRPipeline/PreprocessDLRData/output/training/training.parquet"
    val_input = "s3://sagemaker-us-east-1-914326228175/DLRPipeline/PreprocessDLRData/output/validation/validation.parquet"

    train_data = load_data_from_s3(train_input)
    val_data = load_data_from_s3(val_input)

    train_data_df = pd.read_parquet(io.BytesIO(train_data))
    validation_data_df = pd.read_parquet(io.BytesIO(val_data))

    api_key = get_secret("ALPACA_API_KEY")
    api_secret = get_secret("ALPACA_API_SECRET")
    api_url = get_secret("ALPACA_API_BASE_URL")

    train_model(
        train_data=train_data_df, 
        validation_data=validation_data_df,
        api_key=api_key,
        api_secret=api_secret,
        api_url=api_url
    )

    bucket_name = 'rl-trading-v1-runs'
    local_path = 'models/runs/papertrading_erl_retrain/actor.pth'
    save_s3_path = 'runs/actor.pth'

    model = load_model_from_local_path(local_path)

    save_model_to_s3(model, bucket_name, save_s3_path)
