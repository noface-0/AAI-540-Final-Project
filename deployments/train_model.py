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


API_KEY = get_var("API_KEY")
API_SECRET = get_var("API_SECRET")
API_BASE_URL = get_var("API_BASE_URL")



def train_model(data=None):
    # Initialize environment
    env = StockTradingEnv

    agent_configs = {
        "ppo": ERL_PARAMS,
        "sac": ERL_PARAMS
    }
    params = agent_configs.get(AGENT)

    # Training phase
    print("Starting training phase...")
    train(
        data=data,
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
        break_step=1e6
    )
    
    # Testing phase
    print("Starting testing phase...")
    account_value_erl = test(
        data=data,
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
        net_dimension=params['net_dimension']
    )
    print(
        "Testing phase completed. Final account value:", 
        account_value_erl
    )

    print("Starting full data training phase...")
    train(
        data=data,
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
    train_model()