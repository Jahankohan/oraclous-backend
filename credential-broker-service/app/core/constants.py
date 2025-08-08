# Data source capability mapping
DATA_SOURCE_CAPABILITIES = {
    "google": {
        "drive": {
            "required_scopes": ["https://www.googleapis.com/auth/drive.readonly"],
            "operations": ["list_files", "download_file", "get_metadata", "search"],
            "file_types": ["docs", "sheets", "slides", "pdf", "images", "videos", "archives"],
            "supports_streaming": True,
            "supports_webhooks": True
        },
        "docs": {
            "required_scopes": ["https://www.googleapis.com/auth/documents.readonly"],
            "operations": ["read_document", "export_as_text", "export_as_html"],
            "supports_streaming": False,
            "supports_webhooks": False
        },
        "sheets": {
            "required_scopes": ["https://www.googleapis.com/auth/spreadsheets.readonly"],
            "operations": ["read_sheet", "list_sheets", "get_values", "export_csv"],
            "supports_streaming": False,
            "supports_webhooks": False
        }
    },
    "notion": {
        "pages": {
            "required_scopes": [],  # Notion uses workspace permissions
            "operations": ["list_pages", "read_page", "get_blocks", "search"],
            "supports_streaming": False,
            "supports_webhooks": True
        },
        "databases": {
            "required_scopes": [],
            "operations": ["list_databases", "query_database", "get_properties", "export_data"],
            "supports_streaming": False,
            "supports_webhooks": True
        }
    },
    "github": {
        "repositories": {
            "required_scopes": ["repo"],
            "operations": ["list_repos", "get_repo", "list_files", "get_file_content", "search_code"],
            "supports_streaming": False,
            "supports_webhooks": True
        },
        "issues": {
            "required_scopes": ["repo"],
            "operations": ["list_issues", "get_issue", "search_issues"],
            "supports_streaming": False,
            "supports_webhooks": True
        },
        "pull_requests": {
            "required_scopes": ["repo"],
            "operations": ["list_prs", "get_pr", "get_pr_files", "get_pr_comments"],
            "supports_streaming": False,
            "supports_webhooks": True
        }
    }
}

# Common error codes for OAuth operations
OAUTH_ERROR_CODES = {
    "TOKEN_NOT_FOUND": "oauth_token_not_found",
    "TOKEN_EXPIRED": "oauth_token_expired",
    "INSUFFICIENT_SCOPES": "oauth_insufficient_scopes", 
    "REFRESH_FAILED": "oauth_refresh_failed",
    "PROVIDER_ERROR": "oauth_provider_error",
    "INVALID_PROVIDER": "oauth_invalid_provider",
    "RATE_LIMITED": "oauth_rate_limited"
}