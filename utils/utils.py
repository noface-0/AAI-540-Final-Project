import os
from dotenv import load_dotenv

load_dotenv()


def get_var(key) -> str:
    """Retrieve a credential by its key from environment variables."""
    _key = os.getenv(key)
    if not key:
        load_dotenv() # loading again for new variables from sagemaker
        _key = os.getenv(key)
    return _key