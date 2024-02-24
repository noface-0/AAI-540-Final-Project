import threading
from fastapi import FastAPI

from config.indicators import INDICATORS
from config.tickers import DOW_30_TICKER
from config.models import ERL_PARAMS, SAC_PARAMS
from config.training import TIME_INTERVAL, AGENT
from environments.alpaca import AlpacaPaperTrading
from utils.utils import get_var

app = FastAPI()

action_dim = len(DOW_30_TICKER)
state_dim = 1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim

API_KEY = get_var("API_KEY")
API_SECRET = get_var("API_SECRET")
API_BASE_URL = get_var("API_BASE_URL")


def start_trading():
    agent_configs = {
        "ppo": ERL_PARAMS,
        "sac": SAC_PARAMS
    }
    params = agent_configs.get(AGENT)

    paper_trading_erl = AlpacaPaperTrading(
        ticker_list=DOW_30_TICKER,
        time_interval=TIME_INTERVAL,
        drl_lib='elegantrl',
        agent=AGENT,
        cwd='models/runs/papertrading_erl_retrain',
        net_dim=params['net_dimension'],
        state_dim=state_dim,
        action_dim=action_dim,
        API_KEY=API_KEY,
        API_SECRET=API_SECRET,
        API_BASE_URL=API_BASE_URL,
        tech_indicator_list=INDICATORS,
        turbulence_thresh=30,
        max_stock=1e2
    )
    paper_trading_erl.run()


@app.on_event("startup")
def on_startup():
    thread = threading.Thread(target=start_trading)
    thread.start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)