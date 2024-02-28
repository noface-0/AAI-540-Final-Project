import json
import logging
import os

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# evaluation already performed. Just extracting for Sagemaker step
if __name__ == "__main__":
    evaluation_json_path = f'{BASE_DIR}/models/runs/eval/evaluation.json'

    with open(evaluation_json_path, 'r') as f:
        evaluation_data = json.load(f)

    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    print("Saving return report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(evaluation_data))