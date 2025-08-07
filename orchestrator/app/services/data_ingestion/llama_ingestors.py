import os
import asyncio
from typing import List, Dict, Any, Optional
import uuid
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.readers.google import GoogleDriveReader
from llama_index.readers.notion import NotionPageReader
from llama_index.readers.database import DatabaseReader
import httpx

from .base import BaseIngestor, DataSource, IngestionJob, IngestionResult, IngestionStatus
import logging

logger = logging.getLogger(__name__)

class LlamaIndexGoogleDriveIngestor(BaseIngestor):
    """Google Drive ingestor using LlamaIndex"""
    
    def __init__(self):
        super().__init__("llama_google_drive", ["google_drive"])
    
    async def validate_config(self, config: Dict[str, Any]) -> bool:
        required_fields = ["folder_id"]
        return all(field in config for field in required_fields)
    
    async def test_connection(self, source: DataSource) -> bool:
        try:
            # Get OAuth token from auth service
            token_data = await self._get_oauth_token(source.credentials.get("user_id"))
            if not token_data:
                return False
            
            # Test by trying to list files in the folder
            reader = GoogleDriveReader()
            # This would require the actual token - simplified for now
            return True
        except Exception as e:
            logger.error(f"Google Drive connection test failed: {e}")
            return False
    
    async def get_available_resources(self, source: DataSource) -> List[Dict[str, Any]]:
        try:
            token_data = await self._get_oauth_token(source.credentials.get("user_id"))
            if not token_data:
                return []
            
            # Use Google Drive API to list files
            headers = {"Authorization": f"Bearer {token_data['access_token']}"}
            folder_id = source.config.get("folder_id")
            
            async with httpx.AsyncClient() as client:
                url = f"https://www.googleapis.com/drive/v3/files"
                params = {
                    "q": f"'{folder_id}' in parents" if folder_id != "root" else None,
                    "fields": "files(id,name,mimeType,size,modifiedTime)"
                }
                params = {k: v for k, v in params.items() if v is not None}
                
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                files_data = response.json()
                return [
                    {
                        "id": file["id"],
                        "name": file["name"],
                        "type": file["mimeType"],
                        "size": file.get("size"),
                        "modified": file.get("modifiedTime")
                    }
                    for file in files_data.get("files", [])
                ]
        except Exception as e:
            logger.error(f"Failed to list Google Drive resources: {e}")
            return []
    
    async def ingest(self, source: DataSource, job: IngestionJob) -> IngestionResult:
        try:
            token_data = await self._get_oauth_token(source.credentials.get("user_id"))
            if not token_data:
                raise Exception("Failed to get OAuth token")
            
            # Configure the reader
            reader = GoogleDriveReader()
            
            # Run the ingestion in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(
                None,
                self._load_documents,
                reader,
                source.config,
                token_data
            )
            
            # Convert LlamaIndex documents to our format
            doc_data = []
            for doc in documents:
                doc_data.append({
                    "id": doc.id_ or str(uuid.uuid4()),
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "source": "google_drive"
                })
            
            return IngestionResult(
                job_id=job.id,
                documents=doc_data,
                job_metadata={
                    "source_type": "google_drive",
                    "folder_id": source.config.get("folder_id"),
                    "processed_count": len(doc_data)
                }
            )
            
        except Exception as e:
            logger.error(f"Google Drive ingestion failed: {e}")
            return IngestionResult(
                job_id=job.id,
                documents=[],
                job_metadata={},
                errors=[str(e)]
            )
    
    def _load_documents(self, reader, config, token_data):
        """Load documents synchronously"""
        folder_id = config.get("folder_id", "root")
        file_types = config.get("file_types", [])
        
        # Set up authentication for the reader
        # This would require configuring the reader with the token
        
        return reader.load_data(folder_id=folder_id)
    
    async def _get_oauth_token(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get OAuth token from auth service"""
        try:
            auth_service_url = os.getenv("AUTH_SERVICE_URL", "http://auth-service:80")
            internal_key = os.getenv("INTERNAL_SERVICE_KEY", "your_internal_service_key")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{auth_service_url}/oauth/runtime-tokens",
                    params={"user_id": user_id, "provider": "google"},
                    headers={"X-Internal-Key": internal_key}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get OAuth token: {e}")
            return None

class LlamaIndexNotionIngestor(BaseIngestor):
    """Notion ingestor using LlamaIndex"""
    
    def __init__(self):
        super().__init__("llama_notion", ["notion"])
    
    async def validate_config(self, config: Dict[str, Any]) -> bool:
        # Can work with page_ids or database_id
        return "page_ids" in config or "database_id" in config
    
    async def test_connection(self, source: DataSource) -> bool:
        try:
            token_data = await self._get_oauth_token(source.credentials.get("user_id"))
            if not token_data:
                return False
            
            # Test by making a simple API call
            headers = {
                "Authorization": f"Bearer {token_data['access_token']}",
                "Notion-Version": "2022-06-28"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.notion.com/v1/users/me",
                    headers=headers
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Notion connection test failed: {e}")
            return False
    
    async def get_available_resources(self, source: DataSource) -> List[Dict[str, Any]]:
        try:
            token_data = await self._get_oauth_token(source.credentials.get("user_id"))
            if not token_data:
                return []
            
            headers = {
                "Authorization": f"Bearer {token_data['access_token']}",
                "Notion-Version": "2022-06-28"
            }
            
            resources = []
            
            # If database_id is provided, list database pages
            if "database_id" in source.config:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"https://api.notion.com/v1/databases/{source.config['database_id']}/query",
                        headers=headers,
                        json={"page_size": 100}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        for page in data.get("results", []):
                            resources.append({
                                "id": page["id"],
                                "name": page.get("properties", {}).get("Name", {}).get("title", [{}])[0].get("plain_text", "Untitled"),
                                "type": "page",
                                "url": page["url"],
                                "last_edited": page["last_edited_time"]
                            })
            
            return resources
        except Exception as e:
            logger.error(f"Failed to list Notion resources: {e}")
            return []
    
    async def ingest(self, source: DataSource, job: IngestionJob) -> IngestionResult:
        try:
            token_data = await self._get_oauth_token(source.credentials.get("user_id"))
            if not token_data:
                raise Exception("Failed to get OAuth token")
            
            # Configure the reader
            integration_token = token_data['access_token']
            
            # Run the ingestion in a thread pool
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(
                None,
                self._load_notion_documents,
                integration_token,
                source.config
            )
            
            # Convert to our format
            doc_data = []
            for doc in documents:
                doc_data.append({
                    "id": doc.id_ or str(uuid.uuid4()),
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "source": "notion"
                })
            
            return IngestionResult(
                job_id=job.id,
                documents=doc_data,
                job_metadata={
                    "source_type": "notion",
                    "processed_count": len(doc_data)
                }
            )
            
        except Exception as e:
            logger.error(f"Notion ingestion failed: {e}")
            return IngestionResult(
                job_id=job.id,
                documents=[],
                job_metadata={},
                errors=[str(e)]
            )
    
    def _load_notion_documents(self, integration_token, config):
        """Load Notion documents synchronously"""
        reader = NotionPageReader(integration_token=integration_token)
        
        if "page_ids" in config:
            return reader.load_data(page_ids=config["page_ids"])
        elif "database_id" in config:
            return reader.load_data(database_id=config["database_id"])
        else:
            raise ValueError("Either page_ids or database_id must be provided")
    
    async def _get_oauth_token(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get OAuth token from auth service"""
        try:
            auth_service_url = os.getenv("AUTH_SERVICE_URL", "http://auth-service:80")
            internal_key = os.getenv("INTERNAL_SERVICE_KEY", "your_internal_service_key")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{auth_service_url}/oauth/runtime-tokens",
                    params={"user_id": user_id, "provider": "notion"},
                    headers={"X-Internal-Key": internal_key}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get OAuth token: {e}")
            return None

class LlamaIndexDatabaseIngestor(BaseIngestor):
    """Database ingestor using LlamaIndex"""
    
    def __init__(self):
        super().__init__("llama_database", ["database", "postgresql", "mysql", "sqlite"])
    
    async def validate_config(self, config: Dict[str, Any]) -> bool:
        required_fields = ["connection_string", "query"]
        return all(field in config for field in required_fields)
    
    async def test_connection(self, source: DataSource) -> bool:
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._test_db_connection,
                source.config["connection_string"]
            )
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    def _test_db_connection(self, connection_string: str) -> bool:
        """Test database connection synchronously"""
        from sqlalchemy import create_engine, text
        try:
            engine = create_engine(connection_string)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
    
    async def get_available_resources(self, source: DataSource) -> List[Dict[str, Any]]:
        """For databases, we could list tables or return query examples"""
        try:
            loop = asyncio.get_event_loop()
            tables = await loop.run_in_executor(
                None,
                self._list_tables,
                source.config["connection_string"]
            )
            return [{"name": table, "type": "table"} for table in tables]
        except Exception as e:
            logger.error(f"Failed to list database resources: {e}")
            return []
    
    def _list_tables(self, connection_string: str) -> List[str]:
        """List database tables"""
        from sqlalchemy import create_engine, inspect
        try:
            engine = create_engine(connection_string)
            inspector = inspect(engine)
            return inspector.get_table_names()
        except Exception:
            return []
    
    async def ingest(self, source: DataSource, job: IngestionJob) -> IngestionResult:
        try:
            # Run database query in thread pool
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(
                None,
                self._load_database_documents,
                source.config
            )
            
            # Convert to our format
            doc_data = []
            for doc in documents:
                doc_data.append({
                    "id": doc.id_ or str(uuid.uuid4()),
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "source": "database"
                })
            
            return IngestionResult(
                job_id=job.id,
                documents=doc_data,
                job_metadata={
                    "source_type": "database",
                    "processed_count": len(doc_data)
                }
            )
            
        except Exception as e:
            logger.error(f"Database ingestion failed: {e}")
            return IngestionResult(
                job_id=job.id,
                documents=[],
                job_metadata={},
                errors=[str(e)]
            )
    
    def _load_database_documents(self, config):
        """Load database documents synchronously"""
        reader = DatabaseReader(
            engine=config["connection_string"],
            query=config["query"]
        )
        return reader.load_data()
