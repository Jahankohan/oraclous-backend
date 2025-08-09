from typing import Dict, Any, Optional
from app.tools.base.internal_tool import InternalTool
from app.schemas.tool_instance import ExecutionContext


class OAuthTool(InternalTool):
    """
    Base class for tools that require OAuth authentication
    """
    
    async def validate_context(self, context: ExecutionContext) -> bool:
        """Validate OAuth credentials"""
        if not await super().validate_context(context):
            return False
        
        oauth_creds = self.get_credentials(context, "OAUTH_TOKEN")
        if not oauth_creds:
            return False
        
        # Check if token is present and not expired
        access_token = oauth_creds.get("access_token")
        expires_at = oauth_creds.get("expires_at")
        
        if not access_token:
            return False
        
        # If expires_at is provided, check if token is still valid
        if expires_at:
            from datetime import datetime
            if datetime.fromisoformat(expires_at) <= datetime.utcnow():
                return False
        
        return True
    
    def get_oauth_headers(self, context: ExecutionContext) -> Dict[str, str]:
        """Get OAuth authorization headers"""
        oauth_creds = self.get_credentials(context, "OAUTH_TOKEN")
        if not oauth_creds:
            raise ValueError("OAuth credentials not found")
        
        access_token = oauth_creds["access_token"]
        return {"Authorization": f"Bearer {access_token}"}

