from typing import Any, Dict
import io
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from app.tools.base.auth_tool import OAuthTool
from app.schemas.tool_instance import ExecutionContext, ExecutionResult
from app.schemas.tool_definition import (
    ToolDefinition, ToolSchema, ToolCapability, CredentialRequirement
)
from app.schemas.common import ToolCategory, ToolType, CredentialType


class GoogleDriveReader(OAuthTool):
    """
    Tool for reading files from Google Drive
    Supports various file types: CSV, Excel, Docs, Sheets, etc.
    """
    
    @classmethod
    def get_tool_definition(cls) -> ToolDefinition:
        """Return the tool definition for Google Drive Reader"""
        return ToolDefinition(
            name="Google Drive Reader",
            description="Read and extract data from Google Drive files including Sheets, Docs, CSV, and Excel files",
            version="1.0.0",
            category=ToolCategory.INGESTION,
            type=ToolType.INTERNAL,
            capabilities=[
                ToolCapability(
                    name="read_drive_files",
                    description="Read files from Google Drive"
                ),
                ToolCapability(
                    name="list_drive_files", 
                    description="List files in Google Drive folders"
                ),
                ToolCapability(
                    name="extract_spreadsheet_data",
                    description="Extract data from Google Sheets"
                ),
                ToolCapability(
                    name="download_files",
                    description="Download files from Google Drive"
                )
            ],
            tags=["google", "drive", "spreadsheet", "document", "cloud", "ingestion"],
            input_schema=ToolSchema(
                type="object",
                properties={
                    "file_id": {
                        "type": "string",
                        "description": "Google Drive file ID"
                    },
                    "file_path": {
                        "type": "string", 
                        "description": "Path to file in Google Drive (alternative to file_id)"
                    },
                    "sheet_name": {
                        "type": "string",
                        "description": "Sheet name for Google Sheets files (optional)"
                    },
                    "range": {
                        "type": "string",
                        "description": "Cell range for Google Sheets (e.g., 'A1:Z100')"
                    },
                    "include_headers": {
                        "type": "boolean",
                        "description": "Whether to include headers (default: true)"
                    },
                    "file_type": {
                        "type": "string", 
                        "enum": ["auto", "sheets", "csv", "excel", "docs"],
                        "description": "File type to process (auto-detect if not specified)"
                    }
                },
                required=["file_id"],
                description="Input parameters for Google Drive file reading"
            ),
            output_schema=ToolSchema(
                type="object",
                properties={
                    "data": {
                        "type": "array",
                        "description": "Extracted data rows"
                    },
                    "headers": {
                        "type": "array",
                        "description": "Column headers"
                    },
                    "file_info": {
                        "type": "object",
                        "description": "File metadata information"
                    },
                    "row_count": {
                        "type": "integer",
                        "description": "Number of data rows"
                    }
                },
                description="Extracted file data and metadata"
            ),
            credential_requirements=[
                CredentialRequirement(
                    type=CredentialType.OAUTH_TOKEN,
                    required=True,
                    scopes=["https://www.googleapis.com/auth/drive.readonly"],
                    description="Google OAuth token with Drive read access"
                )
            ]
        )
    
    async def _execute_internal(
        self, 
        input_data: Dict[str, Any], 
        context: ExecutionContext
    ) -> ExecutionResult:
        """Execute Google Drive file reading"""
        try:
            # Get OAuth credentials
            oauth_creds = self.get_credentials(context, "OAUTH_TOKEN")
            access_token = oauth_creds["access_token"]
            
            # Build Google Drive service
            from google.oauth2.credentials import Credentials
            credentials = Credentials(token=access_token)
            drive_service = build('drive', 'v3', credentials=credentials)
            
            file_id = input_data["file_id"]
            file_type = input_data.get("file_type", "auto")
            
            # Get file metadata
            file_metadata = drive_service.files().get(
                fileId=file_id,
                fields="id,name,mimeType,size,modifiedTime"
            ).execute()
            
            # Determine processing method based on file type
            if file_type == "auto":
                file_type = self._detect_file_type(file_metadata["mimeType"])
            
            # Process based on file type
            if file_type == "sheets":
                result_data = await self._process_google_sheets(
                    file_id, input_data, credentials
                )
            elif file_type in ["csv", "excel"]:
                result_data = await self._process_file_download(
                    drive_service, file_id, file_type, input_data
                )
            elif file_type == "docs":
                result_data = await self._process_google_docs(
                    file_id, input_data, credentials
                )
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            return ExecutionResult(
                success=True,
                data={
                    **result_data,
                    "file_info": {
                        "id": file_metadata["id"],
                        "name": file_metadata["name"],
                        "mime_type": file_metadata["mimeType"],
                        "size": file_metadata.get("size"),
                        "modified_time": file_metadata["modifiedTime"]
                    }
                },
                metadata={
                    "file_type": file_type,
                    "processing_time": "calculated_later"
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                error_message=f"Failed to read Google Drive file: {str(e)}"
            )
    
    async def _list_drive_files(
        self,
        input_data: Dict[str, Any],
        credentials
    ) -> Dict[str, Any]:
        """
        List files in a Google Drive folder.
        input_data can include 'folder_id' (optional), 'query' (optional), 'page_size' (optional)
        """
        drive_service = build('drive', 'v3', credentials=credentials)
        folder_id = input_data.get("folder_id")
        query = input_data.get("query")
        page_size = input_data.get("page_size", 100)

        # Build query for files().list()
        q = []
        if folder_id:
            q.append(f"'{folder_id}' in parents")
        if query:
            q.append(query)
        query_str = " and ".join(q) if q else None

        results = drive_service.files().list(
            q=query_str,
            pageSize=page_size,
            fields="files(id, name, mimeType, modifiedTime, size)"
        ).execute()

        files = results.get("files", [])
        data = [[f["id"], f["name"], f["mimeType"], f.get("size"), f["modifiedTime"]] for f in files]
        headers = ["id", "name", "mimeType", "size", "modifiedTime"]

        return {
            "data": data,
            "headers": headers,
            "row_count": len(data),
            "metadata": {"operation": "list_drive_files"}
        }
    
    def _detect_file_type(self, mime_type: str) -> str:
        """Detect file type from MIME type"""
        mime_mapping = {
            "application/vnd.google-apps.spreadsheet": "sheets",
            "application/vnd.google-apps.document": "docs", 
            "text/csv": "csv",
            "application/vnd.ms-excel": "excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "excel"
        }
        return mime_mapping.get(mime_type, "unknown")
    
    async def _process_google_sheets(
        self, 
        file_id: str, 
        input_data: Dict[str, Any],
        credentials
    ) -> Dict[str, Any]:
        """Process Google Sheets file"""
        sheets_service = build('sheets', 'v4', credentials=credentials)
        
        sheet_name = input_data.get("sheet_name")
        cell_range = input_data.get("range", "")
        include_headers = input_data.get("include_headers", True)
        
        # Build range string
        range_string = sheet_name if sheet_name else "Sheet1"
        if cell_range:
            range_string += f"!{cell_range}"
        
        # Get spreadsheet data
        result = sheets_service.spreadsheets().values().get(
            spreadsheetId=file_id,
            range=range_string
        ).execute()
        
        values = result.get('values', [])
        if not values:
            return {"data": [], "headers": [], "row_count": 0}
        
        headers = []
        data = values
        
        if include_headers and values:
            headers = values[0]
            data = values[1:]
        
        return {
            "data": data,
            "headers": headers,
            "row_count": len(data)
        }
    
    async def _process_file_download(
        self, 
        drive_service, 
        file_id: str, 
        file_type: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process downloadable files (CSV, Excel)"""
        # Download file content
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        fh.seek(0)
        
        # Process based on file type
        if file_type == "csv":
            df = pd.read_csv(fh)
        elif file_type == "excel":
            df = pd.read_excel(fh)
        else:
            raise ValueError(f"Unsupported file type for download: {file_type}")
        
        include_headers = input_data.get("include_headers", True)
        
        if include_headers:
            headers = df.columns.tolist()
            data = df.values.tolist()
        else:
            headers = []
            data = df.values.tolist()
        
        return {
            "data": data,
            "headers": headers,
            "row_count": len(data)
        }
    
    async def _process_google_docs(
        self,
        file_id: str,
        input_data: Dict[str, Any], 
        credentials
    ) -> Dict[str, Any]:
        """Process Google Docs file (extract text content)"""
        docs_service = build('docs', 'v1', credentials=credentials)
        
        # Get document content
        document = docs_service.documents().get(documentId=file_id).execute()
        
        # Extract text content
        content = []
        for element in document.get('body', {}).get('content', []):
            if 'paragraph' in element:
                paragraph = element['paragraph']
                for text_element in paragraph.get('elements', []):
                    if 'textRun' in text_element:
                        content.append(text_element['textRun']['content'])
        
        full_text = ''.join(content)
        
        # Split into lines for structured output
        lines = [line.strip() for line in full_text.split('\n') if line.strip()]
        
        return {
            "data": [[line] for line in lines],  # Each line as a row
            "headers": ["content"],
            "row_count": len(lines)
        }
    
    def calculate_credits(self, input_data: Any, result: ExecutionResult) -> float:
        """Calculate credits based on data processed"""
        if not result.success or not result.data:
            return 0.1  # Minimal charge for failed attempts
        
        row_count = result.data.get("row_count", 0)
        
        # Credit calculation: 0.1 base + 0.001 per row
        base_credits = 0.1
        row_credits = row_count * 0.001
        
        return base_credits + row_credits
