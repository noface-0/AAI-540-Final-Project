import os
import argparse
import pandas as pd

from environments.base import StockTradingEnv
from training.train_test import test
from config.indicators import INDICATORS
from config.tickers import DOW_30_TICKER
from config.models import ERL_PARAMS, SAC_PARAMS
from config.training import (
    TIME_INTERVAL,
    TEST_START_DATE,
    TEST_END_DATE,
    AGENT
)
from deployments.s3_utils import (
    get_secret
)


def test_model(
        test_data=pd.DataFrame(), 
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
    
    # Testing phase
    print("Starting testing phase...")
    account_value_erl = test(
        data=test_data,
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        time_interval=TIME_INTERVAL,
        technical_indicator_list=INDICATORS,
        drl_lib='elegantrl',
        env=env,
        model_name=AGENT,
        if_vix=True,
        cwd=f'{BASE_DIR}/models/runs/papertrading_erl',
        net_dimension=params['net_dimension'],
        split=False,
        api_key=api_key,
        api_secret=api_secret,
        api_url=api_url
    )
    print(
        "Testing phase completed. Final account value:", 
        account_value_erl
    )


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description="Process input data for testing.")
    parser.add_argument('--testing', type=str, default=os.environ.get('S3_TESTING'))
    args = parser.parse_args()

    # test_input = os.environ.get('S3_TESTING')
    # test_data = load_data_from_s3(test_input)
    # test_data_df = pd.read_parquet(io.BytesIO(test_data))

    api_key = get_secret("ALPACA_API_KEY")
    api_secret = get_secret("ALPACA_API_SECRET")
    api_url = get_secret("ALPACA_API_BASE_URL")

    test_model(
        # test_data=test_data_df, 
        api_key=api_key,
        api_secret=api_secret,
        api_url=api_url
    )

