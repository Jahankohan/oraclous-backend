import os
import httpx
from typing import List, Optional
from urllib.parse import urlencode
from datetime import datetime, timedelta

from app.core.constants import PROVIDERS
from app.core.jwt_handler import sign_state
from app.repositories.token_repository import TokenRepository
from app.schema.oauth_schemas import EnsureAccessResponse


class OAuthService:
    def __init__(self, repository: TokenRepository):
        self.repository = repository

    async def build_login_url(self, provider: str, state: str, required_scopes: Optional[List[str]] = None) -> str:
        """Create OAuth login URL with signed state."""
        config = PROVIDERS[provider]
        scopes = required_scopes or config["scopes"]

        client_id = os.getenv(f"{provider.upper()}_CLIENT_ID")
        redirect_uri = f"{os.getenv('REDIRECT_URI')}/oauth/{provider}/callback"

        signed_state = sign_state({"state": state, "provider": provider})

        query = urlencode({
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes),
            "state": signed_state
        })

        return f"{config['authorize_url']}?{query}"

    async def exchange_token(self, provider: str, code: str):
        """Exchange authorization code for access/refresh token."""
        config = PROVIDERS[provider]
        auth = None
        data = {
            "code": code,
            "redirect_uri": f"{os.getenv('REDIRECT_URI')}/oauth/{provider}/callback",
        }

        if provider == "google":
            data["grant_type"] = "authorization_code"
            data["client_id"] = os.getenv(f"{provider.upper()}_CLIENT_ID")
            data["client_secret"] = os.getenv(f"{provider.upper()}_CLIENT_SECRET")
        
        if provider == "notion":
            data["grant_type"] = "authorization_code"
            auth = (os.getenv(f"{provider.upper()}_CLIENT_ID"), os.getenv(f"{provider.upper()}_CLIENT_SECRET"))

        headers = {}
        if provider == "github":
            headers["Accept"] = "application/json"
            auth = (os.getenv(f"{provider.upper()}_CLIENT_ID"), os.getenv(f"{provider.upper()}_CLIENT_SECRET"))
    
        print("Exchanging New:", data)
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                config["token_url"],
                data=data,
                headers=headers,
                auth=auth
            )
            resp.raise_for_status()
            token_data = resp.json()

        expires_in = token_data.get("expires_in", 3600)
        expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

        return {
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token"),
            "scopes": token_data.get("scope", "").split() if "scope" in token_data else [],
            "expires_at": expires_at
        }

    async def refresh_token(self, user_id: str, provider: str, refresh_token: str) -> dict:
        """Refresh access token using refresh token."""
        config = PROVIDERS[provider]

        data = {
            "client_id": os.getenv(f"{provider.upper()}_CLIENT_ID"),
            "client_secret": os.getenv(f"{provider.upper()}_CLIENT_SECRET"),
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }

        headers = {"Accept": "application/json"}
        auth = None
        
        # Provider-specific auth configuration
        if provider in ["github", "notion"]:
            auth = (os.getenv(f"{provider.upper()}_CLIENT_ID"), os.getenv(f"{provider.upper()}_CLIENT_SECRET"))

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                config["token_url"],
                data=data,
                headers=headers,
                auth=auth
            )
            resp.raise_for_status()
            token_data = resp.json()

        access_token = token_data.get("access_token")
        new_refresh_token = token_data.get("refresh_token", refresh_token)  # Some providers don't return new refresh token
        expires_in = token_data.get("expires_in", 3600)
        expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

        # Update token in database
        await self.repository.update_access_token(user_id, provider, access_token, expires_at)
        
        # If we got a new refresh token, update that too
        if new_refresh_token != refresh_token:
            token_obj = await self.repository.get_token(user_id, provider)
            if token_obj:
                await self.repository.save_token(
                    user_id=user_id,
                    provider=provider,
                    access_token=access_token,
                    refresh_token=new_refresh_token,
                    scopes=token_obj.scopes,
                    expires_at=expires_at
                )

        return {
            "access_token": access_token,
            "expires_at": expires_at,
            "refresh_token": new_refresh_token
        }
    
    async def fetch_user_profile(self, provider: str, access_token: str) -> dict:
        """Fetch user profile information from the provider."""
        if provider == "google":
            url = "https://www.googleapis.com/oauth2/v2/userinfo"
            headers = {"Authorization": f"Bearer {access_token}"}
        elif provider == "github":
            url = "https://api.github.com/user"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/vnd.github+json"
            }
        elif provider == "notion":
            url = "https://api.notion.com/v1/users/me"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Notion-Version": "2022-06-28"
            }
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            profile = resp.json()

            if provider == "github" and not profile.get("email"):
                # Fetch emails if not present
                email_resp = await client.get("https://api.github.com/user/emails", headers=headers)
                email_resp.raise_for_status()
                emails = email_resp.json()

                primary_email = next((e["email"] for e in emails if e.get("primary")), None)
                fallback_email = next((e["email"] for e in emails if e.get("verified")), None)
                profile["email"] = primary_email or fallback_email

        return self._normalize_profile(provider, profile)

    def _normalize_profile(self, provider: str, profile: dict) -> dict:
        """Normalize profile data across providers."""
        if provider == "google":
            return {
                "email": profile.get("email"),
                "first_name": profile.get("given_name"),
                "last_name": profile.get("family_name"),
                "picture": profile.get("picture")
            }
        elif provider == "github":
            full_name = profile.get("name") or ""
            parts = full_name.split(" ", 1)
            return {
                "email": profile.get("email"),
                "first_name": parts[0] if parts else None,
                "last_name": parts[1] if len(parts) > 1 else None,
                "picture": profile.get("avatar_url") or ""
            }
        elif provider == "notion":
            return self.extract_notion_user_info(profile)

    def extract_notion_user_info(self, data: dict):
        """Extract user info from Notion API response."""
        owner_user = data.get("bot", {}).get("owner", {}).get("user", {})

        full_name = owner_user.get("name", "")
        first_name, last_name = None, None
        if full_name:
            parts = full_name.split(" ", 1)
            first_name = parts[0]
            last_name = parts[1] if len(parts) > 1 else None

        return {
            "first_name": first_name,
            "last_name": last_name,
            "email": owner_user.get("person", {}).get("email"),
            "picture": owner_user.get("avatar_url")
        }
    
    async def ensure_access(
        self,
        user_id: str,
        provider: str,
        required_scopes: List[str],
        redirect_state: str
    ) -> dict:
        """
        Ensures that the user has a valid token with the required scopes.
        Returns:
          - {action: 'ok', token: ...}
          - {action: 'reauthenticate', login_url: ..., missing_scopes: ...}
        """
        token_obj = await self.repository.get_token(user_id, provider)

        if not token_obj:
            # No token → start OAuth flow
            login_url = await self.build_login_url(provider, redirect_state, required_scopes)
            return {
                "action": "reauthenticate",
                "login_url": login_url,
                "current_scopes": [],
                "missing_scopes": required_scopes
            }

        # Check for expiry
        if token_obj.expires_at and datetime.utcnow() > token_obj.expires_at:
            if token_obj.refresh_token:
                try:
                    # Attempt to refresh token
                    refreshed = await self.refresh_token(user_id, provider, token_obj.refresh_token)
                    token_obj.access_token = refreshed["access_token"]
                    token_obj.expires_at = refreshed["expires_at"]
                except Exception as e:
                    print(f"Token refresh failed for {provider}: {e}")
                    # Refresh failed → re-authenticate
                    login_url = await self.build_login_url(provider, redirect_state, required_scopes)
                    ensure_access_response = EnsureAccessResponse(
                        action="reauthenticate",
                        login_url=login_url,
                        current_scopes=token_obj.scopes or [],
                        missing_scopes=required_scopes,
                        error="Token refresh failed"
                    )
                    return ensure_access_response
            else:
                # No refresh token → re-authenticate
                login_url = await self.build_login_url(provider, redirect_state, required_scopes)
                ensure_access_response = EnsureAccessResponse(
                    action="reauthenticate",
                    login_url=login_url,
                    current_scopes=token_obj.scopes or [],
                    missing_scopes=required_scopes,
                    error="Token expired, no refresh token"
                )
                return ensure_access_response

        # Check for missing scopes
        current_scopes = set(token_obj.scopes or [])
        missing_scopes = [s for s in required_scopes if s not in current_scopes]

        if missing_scopes:
            # Need additional scopes → re-authenticate with combined scopes
            combined_scopes = list(current_scopes.union(missing_scopes))
            login_url = await self.build_login_url(provider, redirect_state, combined_scopes)
            ensure_access_response = EnsureAccessResponse(
                action="reauthenticate",
                login_url=login_url,
                current_scopes=list(current_scopes) or [],
                missing_scopes=missing_scopes,
                error="insufficient_scopes"
            )
            return ensure_access_response

        # All good
        ensure_access_response = EnsureAccessResponse(
            action="ok",
            token={
                "access_token": token_obj.access_token,
                "expires_at": token_obj.expires_at,
                "scopes": token_obj.scopes
            },
            current_scopes=token_obj.scopes,
            missing_scopes=[],
        )
        return ensure_access_response

    async def validate_token_scopes(self, user_id: str, provider: str, required_scopes: List[str]) -> dict:
        """Validate if user has required scopes without attempting refresh."""
        token_obj = await self.repository.get_token(user_id, provider)
        
        if not token_obj:
            return {
                "valid": False,
                "missing_scopes": required_scopes,
                "current_scopes": [],
                "token_expired": False,
                "token_missing": True
            }

        current_scopes = set(token_obj.scopes or [])
        missing_scopes = [s for s in required_scopes if s not in current_scopes]
        token_expired = token_obj.expires_at and datetime.utcnow() > token_obj.expires_at

        return {
            "valid": len(missing_scopes) == 0 and not token_expired,
            "missing_scopes": missing_scopes,
            "current_scopes": list(current_scopes),
            "token_expired": token_expired,
            "token_missing": False
        }