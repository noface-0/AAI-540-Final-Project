import json
import boto3

# reference: https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-pipelines/tabular/lambda-step/sagemaker-pipelines-lambda-step.ipynb

def endpoint_handler(event, context):
    """ """
    sm_client = boto3.client("sagemaker")

    model_name = event["model_name"]

    endpoint_config_name = event["endpoint_config_name"]
    endpoint_name = event["endpoint_name"]

    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "InstanceType": "ml.m4.xlarge",
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "ModelName": model_name,
                "VariantName": "AllTraffic",
            }
        ],
    )
    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )

    return {
        "statusCode": 200,
        "body": json.dumps("Created Endpoint!"),
        "other_key": "example_value",
    }