import os
from dotenv import load_dotenv
from deployments.s3_utils import get_secret

load_dotenv()


def get_var(key) -> str:
    """Retrieve a credential by its key from environment variables."""
    _key = os.getenv(key)
    if _key is None:
        _key = get_secret(key)
    return _key