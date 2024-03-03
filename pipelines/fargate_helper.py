import json
import boto3


def endpoint_handler(event, context):
    ecs_client = boto3.client('ecs')
    ecr_image_url = event["image_url"]

    task_definition_response = ecs_client.register_task_definition(
        family='rl-trading-system',
        networkMode='awsvpc',
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

    task_definition_arn = task_definition_response \
        ['taskDefinition']['taskDefinitionArn']

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