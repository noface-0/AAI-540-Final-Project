import json
import os
import argparse

from deployments.test_model import test_model
from deployments.s3_utils import (
    get_secret
)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# evaluation already performed. Just extracting for Sagemaker step
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
    evaluation_json_path = f'{BASE_DIR}/models/runs/eval/evaluation.json'

    with open(evaluation_json_path, 'r') as f:
        evaluation_data = json.load(f)

    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    print("Saving return report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(evaluation_data))