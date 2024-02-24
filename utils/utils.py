import os
from dotenv import load_dotenv

load_dotenv()


def get_var(key) -> str:
    """Retrieve a credential by its key from environment variables."""
    return os.getenv(key)