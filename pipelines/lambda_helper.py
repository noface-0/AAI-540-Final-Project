import json
import boto3


def endpoint_handler(event, context):
    ecs_client = boto3.client('ecs')
    ecr_image_url = event["image_url"]

    task_definition_arn = task_definition_response \
        ['taskDefinition']['taskDefinitionArn']

    task_definition_response = ecs_client.register_task_definition(
        family='rl-trading-system',
        networkMode='awsvpc',
        executionRoleArn=task_definition_arn,
        containerDefinitions=[
            {
                'name': 'rl-trading-v1',
                'image': ecr_image_url,
                'essential': True,
                'entryPoint': [
                    "uvicorn", 
                    "deployments/deploy_model:app", 
                    "--host", "0.0.0.0", 
                    "--port", "8080"
                ],
                'portMappings': [
                    {
                        'containerPort': 80,
                        'hostPort': 80,
                        'protocol': 'tcp'
                    },
                ],
                'memory': 512,
                'cpu': 256,
                'logConfiguration': {
                    'logDriver': 'awslogs',
                    'options': {
                        'awslogs-group': '/ecs/rl-trading-system',
                        'awslogs-region': 'us-east-1',
                        'awslogs-stream-prefix': 'ecs'
                    }
                }
            },
        ],
        requiresCompatibilities=['FARGATE'],
        cpu='256',
        memory='512'
    )

    run_task_response = ecs_client.run_task(
        cluster='rl-trading-dev-cluster',
        launchType='FARGATE',
        taskDefinition=task_definition_arn,
        count=1,
        networkConfiguration={
            'awsvpcConfiguration': {
                'subnets': [
                    'subnet-0bd001b0367876f50',
                ],
                'assignPublicIp': 'DISABLED'
            }
        }
    )

    return {
        "statusCode": 200,
        "body": json.dumps("Deployed to Fargate"),
    }


# reference: https://github.com/aws/amazon-sagemaker-examples/blob/main/sagemaker-pipelines/tabular/lambda-step/sagemaker-pipelines-lambda-step.ipynb

def sm_endpoint_handler(event, context):
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