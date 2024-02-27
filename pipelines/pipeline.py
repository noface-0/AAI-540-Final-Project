import os
import json
import boto3
import sagemaker
import sagemaker.session
import awswrangler as wr
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel
from processing.extract import extract_stock_data


BASE_DIR = os.path.dirname(os.path.realpath(__file__))
sagemaker_session = sagemaker.session.Session()
region = sagemaker_session.boto_region_name
role = sagemaker.get_execution_role()
default_bucket = sagemaker_session.default_bucket()
s3_parquet_path = (
    f"s3://{default_bucket}/stock_data/extracted_stock_data.parquet"
)
local_path = "/opt/ml/processing/input/stock_dataset.parquet"


def s3_upload(limit: int=None):
    dataset = extract_stock_data(limit=limit, upload=False)
    wr.s3.to_parquet(
        df=dataset,
        path=s3_parquet_path,
        index=False,
        dataset=True
    )
    print(f"Stock data uploaded to: {s3_parquet_path}")
    dataset.to_parquet(local_path)
    print(f"Stock data saved to: {local_path}")


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.
    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts
    Returns:
        sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")

    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline(
    region,
    project_name=None,
    model_package_group_name="DLRPackageGroup",
    pipeline_name="DLRPipeline",
    base_job_prefix="DLR",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.
    Args:
        region: AWS region to create and run the pipeline.
        processing_role: IAM role to create and run processing steps
        training_role: IAM role to create and run training steps
        data_bucket: the bucket to use for storing the artifacts
    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=s3_parquet_path,
    )

    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-preprocess",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    step_process = ProcessingStep(
        name="PreprocessDLRData",
        processor=sklearn_processor,
        outputs=[
            ProcessingOutput(
                output_name="training",
                source="/opt/ml/processing/training",
                destination=(
                    f"s3://{default_bucket}/{pipeline_name}/"
                    "PreprocessDLRData/output/training"
                )
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/validation",
                destination=(
                    f"s3://{default_bucket}/{pipeline_name}/"
                    "PreprocessDLRData/output/validation"
                )
            ),
            ProcessingOutput(
                output_name="testing",
                source="/opt/ml/processing/testing",
                destination=(
                    f"s3://{default_bucket}/{pipeline_name}/"
                    "PreprocessDLRData/output/testing"
                )
            ),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=["--input-data", input_data]
    )

    model_path = f"s3://{default_bucket}/{base_job_prefix}/DLRModelTrain"

    rl_train = Estimator(
        image_uri='914326228175.dkr.ecr.us-east-1.amazonaws.com/rl-trading-v1:train',
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path=model_path,
        sagemaker_session=sagemaker_session,
        role=role,
        container_entry_point=["python", "deployments/train_model.py"]
    )
    step_train = TrainingStep(
        name="DRLModelTrain",
        estimator=rl_train,
        inputs={
            "training": TrainingInput(
                s3_data=step_process.properties.
                ProcessingOutputConfig.Outputs["training"].
                S3Output.S3Uri,
                content_type="application/x-parquet"
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.
                ProcessingOutputConfig.Outputs["validation"].
                S3Output.S3Uri,
                content_type="application/x-parquet"
            )
        }
    )

    step_register = RegisterModel(
        name="RegisterDLRModel",
        estimator=rl_train,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/x-parquet"],
        response_types=["application/x-parquet"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_process, step_train],
        sagemaker_session=sagemaker_session,
    )
    return pipeline