PROVIDERS = {
    "google": {
        "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "user_info": "https://www.googleapis.com/oauth2/v2/userinfo",
        "revoke_url": "https://oauth2.googleapis.com/revoke",
        "scopes": [
            "openid",
            "https://www.googleapis.com/auth/userinfo.email",
            "https://www.googleapis.com/auth/userinfo.profile", 
            "https://www.googleapis.com/auth/drive.readonly"
        ],
        "scope_descriptions": {
            "openid": "Basic OpenID Connect authentication",
            "https://www.googleapis.com/auth/userinfo.email": "Access to user email address",
            "https://www.googleapis.com/auth/userinfo.profile": "Access to user profile information",
            "https://www.googleapis.com/auth/drive.readonly": "Read-only access to Google Drive files",
            "https://www.googleapis.com/auth/drive": "Full access to Google Drive",
            "https://www.googleapis.com/auth/drive.file": "Access to files created by this app",
            "https://www.googleapis.com/auth/drive.metadata.readonly": "Read-only access to file metadata",
            "https://www.googleapis.com/auth/spreadsheets.readonly": "Read-only access to Google Sheets",
            "https://www.googleapis.com/auth/spreadsheets": "Full access to Google Sheets",
            "https://www.googleapis.com/auth/documents.readonly": "Read-only access to Google Docs",
            "https://www.googleapis.com/auth/documents": "Full access to Google Docs"
        },
        "supports_refresh": True
    },
    "github": {
        "authorize_url": "https://github.com/login/oauth/authorize",
        "token_url": "https://github.com/login/oauth/access_token",
        "user_info": "https://api.github.com/user",
        "revoke_url": "https://github.com/settings/connections/applications/{client_id}",
        "scopes": ["read:user", "user:email"],
        "scope_descriptions": {
            "read:user": "Read access to user profile information",
            "user:email": "Access to user email addresses",
            "repo": "Full control of private repositories",
            "public_repo": "Access to public repositories",
            "read:repo_hook": "Read access to repository hooks",
            "write:repo_hook": "Write access to repository hooks",
            "admin:repo_hook": "Admin access to repository hooks",
            "read:org": "Read-only access to organization membership",
            "write:org": "Read and write access to organization membership",
            "admin:org": "Full control of organization",
            "gist": "Create gists",
            "notifications": "Access notifications",
            "read:discussion": "Read team discussions",
            "write:discussion": "Read and write team discussions"
        },
        "supports_refresh": False
    },
    "notion": {
        "authorize_url": "https://api.notion.com/v1/oauth/authorize",
        "token_url": "https://api.notion.com/v1/oauth/token",
        "user_info": "https://api.notion.com/v1/users/me",
        "scopes": [],
        "scope_descriptions": {
            "read_content": "Read access to pages and databases",
            "update_content": "Edit access to pages and databases",
            "insert_content": "Create new pages and database entries"
        },
        "supports_refresh": False,
    }
}
