import io
import json
import logging
import re
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

s3_client = boto3.client(
    "s3",
    endpoint_url=settings.s3_endpoint,
    aws_access_key_id=settings.s3_access_key,
    aws_secret_access_key=settings.s3_secret_key,
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)


def validate_track_id(track_id: str) -> bool:
    """Validate track_id format to prevent path traversal attacks."""
    # Allow only UUID format: alphanumeric and hyphens
    pattern = r'^[a-f0-9\-]{36}$'
    return bool(re.match(pattern, track_id, re.IGNORECASE))


def ensure_buckets():
    """Create required S3 buckets if they don't exist."""
    for bucket in [settings.s3_bucket_tracks, settings.s3_bucket_visuals]:
        try:
            s3_client.head_bucket(Bucket=bucket)
            logger.info(f"Bucket {bucket} exists")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                try:
                    s3_client.create_bucket(Bucket=bucket)
                    logger.info(f"Created bucket {bucket}")
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket {bucket}: {create_error}")
                    raise
            else:
                logger.error(f"Error checking bucket {bucket}: {e}")
                raise


def upload_track(track_id: str, audio_data: bytes, format: str = "mp3") -> str:
    """Upload an audio track to S3 and return the object key."""
    if not validate_track_id(track_id):
        raise ValueError(f"Invalid track_id format: {track_id}")

    if format not in ("mp3", "wav", "flac"):
        raise ValueError(f"Invalid audio format: {format}")

    try:
        key = f"tracks/{track_id}.{format}"
        s3_client.put_object(
            Bucket=settings.s3_bucket_tracks,
            Key=key,
            Body=audio_data,
            ContentType=f"audio/{format}",
        )
        logger.info(f"Uploaded track {track_id}.{format}")
        return key
    except ClientError as e:
        logger.error(f"Failed to upload track {track_id}: {e}")
        raise


def upload_track_metadata(track_id: str, metadata: dict) -> str:
    """Upload track metadata JSON to S3."""
    if not validate_track_id(track_id):
        raise ValueError(f"Invalid track_id format: {track_id}")

    try:
        key = f"metadata/{track_id}.json"
        s3_client.put_object(
            Bucket=settings.s3_bucket_tracks,
            Key=key,
            Body=json.dumps(metadata, indent=2).encode(),
            ContentType="application/json",
        )
        logger.info(f"Uploaded metadata for track {track_id}")
        return key
    except (ClientError, TypeError, ValueError) as e:
        logger.error(f"Failed to upload metadata for track {track_id}: {e}")
        raise


def download_track(track_id: str, format: str = "mp3") -> bytes:
    """Download an audio track from S3."""
    if not validate_track_id(track_id):
        raise ValueError(f"Invalid track_id format: {track_id}")

    if format not in ("mp3", "wav", "flac"):
        raise ValueError(f"Invalid audio format: {format}")

    try:
        key = f"tracks/{track_id}.{format}"
        response = s3_client.get_object(Bucket=settings.s3_bucket_tracks, Key=key)
        logger.info(f"Downloaded track {track_id}.{format}")
        return response["Body"].read()
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'NoSuchKey':
            logger.error(f"Track {track_id}.{format} not found")
            raise FileNotFoundError(f"Track {track_id}.{format} not found")
        logger.error(f"Failed to download track {track_id}: {e}")
        raise


def upload_visual(track_id: str, visual_data: bytes, format: str = "mp4") -> str:
    """Upload a visual asset to S3."""
    if not validate_track_id(track_id):
        raise ValueError(f"Invalid track_id format: {track_id}")

    if format not in ("mp4", "webm", "png", "jpg"):
        raise ValueError(f"Invalid visual format: {format}")

    try:
        key = f"visuals/{track_id}.{format}"
        content_type = f"video/{format}" if format in ("mp4", "webm") else f"image/{format}"
        s3_client.put_object(
            Bucket=settings.s3_bucket_visuals,
            Key=key,
            Body=visual_data,
            ContentType=content_type,
        )
        logger.info(f"Uploaded visual {track_id}.{format}")
        return key
    except ClientError as e:
        logger.error(f"Failed to upload visual {track_id}: {e}")
        raise


def get_track_url(track_id: str, format: str = "mp3", expires: int = 3600) -> str:
    """Generate a presigned URL for a track."""
    if not validate_track_id(track_id):
        raise ValueError(f"Invalid track_id format: {track_id}")

    if format not in ("mp3", "wav", "flac"):
        raise ValueError(f"Invalid audio format: {format}")

    try:
        key = f"tracks/{track_id}.{format}"
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket_tracks, "Key": key},
            ExpiresIn=expires,
        )
        logger.info(f"Generated presigned URL for track {track_id}.{format}")
        return url
    except ClientError as e:
        logger.error(f"Failed to generate presigned URL for track {track_id}: {e}")
        raise
