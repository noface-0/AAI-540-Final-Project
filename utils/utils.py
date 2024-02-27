import os
from dotenv import load_dotenv

load_dotenv()


def get_var(key) -> str:
    """Retrieve a credential by its key from environment variables."""
    _key = os.getenv(key)
    if _key is None:
        load_dotenv()  # Attempt to reload .env file
        _key = os.getenv(key)
    return _key