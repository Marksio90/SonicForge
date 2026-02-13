import io
import json
from pathlib import Path

import boto3
from botocore.config import Config

from .config import get_settings

settings = get_settings()

s3_client = boto3.client(
    "s3",
    endpoint_url=settings.s3_endpoint,
    aws_access_key_id=settings.s3_access_key,
    aws_secret_access_key=settings.s3_secret_key,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)


def ensure_buckets():
    """Create required S3 buckets if they don't exist."""
    for bucket in [settings.s3_bucket_tracks, settings.s3_bucket_visuals]:
        try:
            s3_client.head_bucket(Bucket=bucket)
        except Exception:
            s3_client.create_bucket(Bucket=bucket)


def upload_track(track_id: str, audio_data: bytes, format: str = "mp3") -> str:
    """Upload an audio track to S3 and return the object key."""
    key = f"tracks/{track_id}.{format}"
    s3_client.put_object(
        Bucket=settings.s3_bucket_tracks,
        Key=key,
        Body=audio_data,
        ContentType=f"audio/{format}",
    )
    return key


def upload_track_metadata(track_id: str, metadata: dict) -> str:
    """Upload track metadata JSON to S3."""
    key = f"metadata/{track_id}.json"
    s3_client.put_object(
        Bucket=settings.s3_bucket_tracks,
        Key=key,
        Body=json.dumps(metadata, indent=2).encode(),
        ContentType="application/json",
    )
    return key


def download_track(track_id: str, format: str = "mp3") -> bytes:
    """Download an audio track from S3."""
    key = f"tracks/{track_id}.{format}"
    response = s3_client.get_object(Bucket=settings.s3_bucket_tracks, Key=key)
    return response["Body"].read()


def upload_visual(track_id: str, visual_data: bytes, format: str = "mp4") -> str:
    """Upload a visual asset to S3."""
    key = f"visuals/{track_id}.{format}"
    s3_client.put_object(
        Bucket=settings.s3_bucket_visuals,
        Key=key,
        Body=visual_data,
        ContentType=f"video/{format}",
    )
    return key


def get_track_url(track_id: str, format: str = "mp3", expires: int = 3600) -> str:
    """Generate a presigned URL for a track."""
    key = f"tracks/{track_id}.{format}"
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket_tracks, "Key": key},
        ExpiresIn=expires,
    )
