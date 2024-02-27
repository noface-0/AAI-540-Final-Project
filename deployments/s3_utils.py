import boto3
import torch
import io


def load_model_from_s3(bucket_name, s3_path):
    """
    Load a PyTorch model directly from an S3 bucket.
    """
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=s3_path)
    bytestream = io.BytesIO(obj['Body'].read())
    model = torch.load(bytestream)
    return model


def save_model_to_s3(model, bucket_name, s3_path):
    """
    Save a PyTorch model to an S3 bucket.
    """
    byte_stream = io.BytesIO()
    torch.save(model, byte_stream)
    byte_stream.seek(0)
    
    s3 = boto3.client('s3')
    try:
        s3.upload_fileobj(byte_stream, bucket_name, s3_path)
        print(
            f"Model successfully uploaded to s3://{bucket_name}/{s3_path}"
        )
    except Exception as e:
        print(
            f"Failed to upload the model to s3://{bucket_name}/{s3_path}. "
            f"Error: {e}"
        )


def load_data_from_s3(s3_path: str):
    """
    Load data from an S3 path.

    Parameters:
    - s3_path (str): The S3 path to the data, in the format s3://bucket/key

    Returns:
    - data (bytes): The data read from the S3 object.
    """
    path_parts = s3_path.split("/")
    bucket = path_parts[2]
    key = "/".join(path_parts[3:])
    
    s3 = boto3.client('s3')

    obj = s3.get_object(Bucket=bucket, Key=key)

    data = obj['Body'].read()
    
    return data


def load_model_from_local_path(local_path):
    """
    Load a PyTorch model from a local file path.

    Parameters:
    - local_path (str): The local file path to the .pth file.

    Returns:
    - model: The loaded PyTorch model.
    """
    model = torch.load(local_path)
    return model