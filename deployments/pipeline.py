import pandas as pd
import boto3
import awswrangler as wr
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

from processing import extract_stock_data


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
        code="processing/preprocess.py",
    )

    step_process = ProcessingStep(
        name="PreprocessData", 
        step_args=processor_args
    )

    return step_process


if __name__ == "__main__":
    # Extract and upload stock data to S3
    upload_result = s3_upload()

    preprocessing_step = integrate_preprocessing(
        s3_parquet_path, role, pipeline_session
    )