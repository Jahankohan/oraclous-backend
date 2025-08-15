import os
import tempfile
import logging
from typing import List, Dict, Any, Optional, BinaryIO
from pathlib import Path
import hashlib

from fastapi import UploadFile
import boto3
from google.cloud import storage as gcs

from app.config.settings import get_settings
from app.core.exceptions import DocumentProcessingError

logger = logging.getLogger(__name__)

class FileHandler:
    """Base class for file handlers"""
    
    def __init__(self):
        self.settings = get_settings()
    
    async def save_uploaded_files(self, files: List[UploadFile]) -> List[str]:
        """Save uploaded files to temporary directory"""
        saved_files = []
        temp_dir = Path(tempfile.gettempdir()) / "llm_graph_builder"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            for file in files:
                # Generate unique filename to avoid conflicts
                file_hash = hashlib.md5(file.filename.encode()).hexdigest()[:8]
                safe_filename = f"{file_hash}_{file.filename}"
                file_path = temp_dir / safe_filename
                
                # Save file
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                saved_files.append(str(file_path))
                logger.info(f"Saved uploaded file: {file.filename} -> {file_path}")
            
            return saved_files
            
        except Exception as e:
            # Clean up any saved files on error
            for file_path in saved_files:
                try:
                    os.unlink(file_path)
                except Exception:
                    pass
            raise DocumentProcessingError(f"Failed to save uploaded files: {e}")
    
    def cleanup_temp_files(self, file_paths: List[str]) -> None:
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.debug(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")

class S3FileHandler(FileHandler):
    """Handler for S3 files"""
    
    def __init__(self):
        super().__init__()
        self.s3_client = None
        self._initialize_s3_client()
    
    def _initialize_s3_client(self):
        """Initialize S3 client if credentials are available"""
        try:
            if self.settings.aws_access_key_id and self.settings.aws_secret_access_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.settings.aws_access_key_id,
                    aws_secret_access_key=self.settings.aws_secret_access_key,
                    region_name=self.settings.aws_region
                )
                logger.info("S3 client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize S3 client: {e}")
    
    async def list_s3_objects(self, bucket_name: str, prefix: str = "") -> List[Dict[str, Any]]:
        """List objects in S3 bucket"""
        if not self.s3_client:
            raise DocumentProcessingError("S3 client not configured")
        
        try:
            objects = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Filter out directories and non-document files
                        if not obj['Key'].endswith('/') and self._is_supported_file(obj['Key']):
                            objects.append({
                                'key': obj['Key'],
                                'size': obj['Size'],
                                'modified': obj['LastModified'],
                                'bucket': bucket_name
                            })
            
            logger.info(f"Found {len(objects)} objects in S3 bucket {bucket_name}")
            return objects
            
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
            raise DocumentProcessingError(f"Failed to list S3 objects: {e}")
    
    async def download_s3_file(self, bucket_name: str, key: str) -> str:
        """Download S3 file to temporary location"""
        if not self.s3_client:
            raise DocumentProcessingError("S3 client not configured")
        
        try:
            temp_dir = Path(tempfile.gettempdir()) / "llm_graph_builder" / "s3"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Create safe filename
            filename = Path(key).name
            file_hash = hashlib.md5(key.encode()).hexdigest()[:8]
            safe_filename = f"{file_hash}_{filename}"
            local_path = temp_dir / safe_filename
            
            # Download file
            self.s3_client.download_file(bucket_name, key, str(local_path))
            
            logger.info(f"Downloaded S3 file: {key} -> {local_path}")
            return str(local_path)
            
        except Exception as e:
            logger.error(f"Failed to download S3 file {key}: {e}")
            raise DocumentProcessingError(f"Failed to download S3 file: {e}")
    
    def _is_supported_file(self, filename: str) -> bool:
        """Check if file type is supported"""
        supported_extensions = {'.pdf', '.txt', '.docx', '.doc', '.html', '.csv', '.json'}
        return Path(filename).suffix.lower() in supported_extensions

class GCSFileHandler(FileHandler):
    """Handler for Google Cloud Storage files"""
    
    def __init__(self):
        super().__init__()
        self.gcs_client = None
        self._initialize_gcs_client()
    
    def _initialize_gcs_client(self):
        """Initialize GCS client if credentials are available"""
        try:
            # GCS client initialization would depend on service account setup
            # This is a placeholder implementation
            self.gcs_client = gcs.Client()
            logger.info("GCS client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize GCS client: {e}")
    
    async def list_gcs_objects(self, bucket_name: str, prefix: str = "") -> List[Dict[str, Any]]:
        """List objects in GCS bucket"""
        if not self.gcs_client:
            raise DocumentProcessingError("GCS client not configured")
        
        try:
            bucket = self.gcs_client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            
            objects = []
            for blob in blobs:
                if not blob.name.endswith('/') and self._is_supported_file(blob.name):
                    objects.append({
                        'name': blob.name,
                        'size': blob.size,
                        'modified': blob.time_created,
                        'bucket': bucket_name
                    })
            
            logger.info(f"Found {len(objects)} objects in GCS bucket {bucket_name}")
            return objects
            
        except Exception as e:
            logger.error(f"Failed to list GCS objects: {e}")
            raise DocumentProcessingError(f"Failed to list GCS objects: {e}")
    
    async def download_gcs_file(self, bucket_name: str, blob_name: str) -> str:
        """Download GCS file to temporary location"""
        if not self.gcs_client:
            raise DocumentProcessingError("GCS client not configured")
        
        try:
            temp_dir = Path(tempfile.gettempdir()) / "llm_graph_builder" / "gcs"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Create safe filename
            filename = Path(blob_name).name
            file_hash = hashlib.md5(blob_name.encode()).hexdigest()[:8]
            safe_filename = f"{file_hash}_{filename}"
            local_path = temp_dir / safe_filename
            
            # Download file
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(str(local_path))
            
            logger.info(f"Downloaded GCS file: {blob_name} -> {local_path}")
            return str(local_path)
            
        except Exception as e:
            logger.error(f"Failed to download GCS file {blob_name}: {e}")
            raise DocumentProcessingError(f"Failed to download GCS file: {e}")
    
    def _is_supported_file(self, filename: str) -> bool:
        """Check if file type is supported"""
        supported_extensions = {'.pdf', '.txt', '.docx', '.doc', '.html', '.csv', '.json'}
        return Path(filename).suffix.lower() in supported_extensions

class FileHandlerFactory:
    """Factory for creating file handlers"""
    
    @staticmethod
    def get_file_handler() -> FileHandler:
        """Get basic file handler"""
        return FileHandler()
    
    @staticmethod
    def get_s3_handler() -> S3FileHandler:
        """Get S3 file handler"""
        return S3FileHandler()
    
    @staticmethod
    def get_gcs_handler() -> GCSFileHandler:
        """Get GCS file handler"""
        return GCSFileHandler()
    
    @staticmethod
    def get_handler_by_source(source_type: str):
        """Get appropriate handler by source type"""
        if source_type.lower() == 's3':
            return FileHandlerFactory.get_s3_handler()
        elif source_type.lower() == 'gcs':
            return FileHandlerFactory.get_gcs_handler()
        else:
            return FileHandlerFactory.get_file_handler()