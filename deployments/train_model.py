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
    save_model_to_s3
)


API_KEY = get_var("API_KEY")
API_SECRET = get_var("API_SECRET")
API_BASE_URL = get_var("API_BASE_URL")


def train_model(train_data=None, validation_data=None):
    # Initialize environment
    env = StockTradingEnv

    agent_configs = {
        "ppo": ERL_PARAMS,
        "sac": ERL_PARAMS
    }
    params = agent_configs.get(AGENT)

    split = not (train_data and validation_data)

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
        split=split
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
        split=split
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
        split=False
    )


if __name__ == "__main__":
    import os
    print(os.environ)
    parser = argparse.ArgumentParser(description="Process input data for training.")
    parser.add_argument('--training', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    args, _ = parser.parse_known_args()

    logging.info(args)

    train_input = args.training
    val_input = args.validation

    train_data = load_data_from_s3(train_input)
    val_data = load_data_from_s3(val_input)

    train_data_df = pd.read_parquet(io.BytesIO(train_data))
    validation_data_df = pd.read_parquet(io.BytesIO(val_data))

    train_model(train_data=train_data_df, validation_data=validation_data_df)

    bucket_name = 'rl-trading-v1-runs'
    local_path = 'models/runs/papertrading_erl_retrain/actor.pth'
    save_s3_path = 'runs/actor.pth'

    model = load_model_from_local_path(local_path)

    save_model_to_s3(model, bucket_name, save_s3_path)

