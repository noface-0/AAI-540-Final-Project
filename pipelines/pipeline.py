import pandas as pd
import boto3
import awswrangler as wr
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.inputs import TrainingInput
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel

from processing.extract import extract_stock_data


sagemaker_session = sagemaker.session.Session()
region = sagemaker_session.boto_region_name
role = sagemaker.get_execution_role()
pipeline_session = PipelineSession()
default_bucket = sagemaker_session.default_bucket()

s3_parquet_path = (
    f"s3://{default_bucket}/stock_data/extracted_stock_data.parquet"
)
local_path = "/opt/ml/processing/input/stock_dataset.parquet"

processing_instance_count = ParameterInteger(
    name="ProcessingInstanceCount", default_value=1
)
instance_type = ParameterString(
    name="TrainingInstanceType", default_value="ml.m5.xlarge"
)
model_approval_status = ParameterString(
    name="ModelApprovalStatus", default_value="PendingManualApproval"
)
batch_data = ParameterString(
    name="BatchData",
    default_value=s3_parquet_path,
)

model_path = f"s3://{default_bucket}/DLRModelTrain"
model_package_name = "DRLModel"
pipeline_name = "DRLPipeline"


def s3_upload():
    dataset = extract_stock_data()

    wr.s3.to_parquet(
        df=dataset,
        path=s3_parquet_path,
        index=False,
        dataset=True
    )
    print(f"Stock data uploaded to: {s3_parquet_path}")

    dataset.to_parquet(local_path)
    print(f"Stock data saved to: {local_path}")


def integrate_preprocessing(s3_parquet_path, role, pipeline_session):
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name="sklearn-preprocess",
        role=role,
        sagemaker_session=pipeline_session,
    )

    processor_args = sklearn_processor.run(
        inputs=[
            ProcessingInput(
                source=batch_data, 
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train", 
                source="/opt/ml/processing/train"
            ),
            ProcessingOutput(
                output_name="validation", 
                source="/opt/ml/processing/validation"
            ),
            ProcessingOutput(
                output_name="test", 
                source="/opt/ml/processing/test"
            ),
        ],
        code="pipelines/preprocess.py",
    )

    step_process = ProcessingStep(
        name="PreprocessData", 
        step_args=processor_args
    )

    return step_process


def integrate_training(step_process):
    rl_train = Estimator(
        image_uri=...,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path=model_path,
        sagemaker_session=pipeline_session,
        role=role,
    )

    train_args = rl_train.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="application/x-parquet"
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="application/x-parquet"
            )
        },
    )

    step_train = TrainingStep(
        name="DRLModelTrain",
        step_args=train_args
    )

    return rl_train, step_train


def integrate_register(rl_train, step_train):
    step_register = RegisterModel(
        name="DRLRegisterModel",
        estimator=rl_train,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/x-parquet"],
        response_types=["application/x-parquet"],
        inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_name,
        approval_status=model_approval_status,
    )
    return step_register


if __name__ == "__main__":
    # Extract and upload stock data to S3
    upload_result = s3_upload()

    step_process = integrate_preprocessing(
        s3_parquet_path, role, pipeline_session
    )
    rl_train, step_train = integrate_training(step_process)

    step_register = integrate_register(rl_train, step_train)

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            model_approval_status,
            batch_data,
        ],
        steps=[step_process, step_train],
    )
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()