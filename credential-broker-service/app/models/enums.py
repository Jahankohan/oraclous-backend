import enum

class CredentialType(str, enum.Enum):
    OAUTH = "oauth"
    API_KEY = "api_key"
    RAW = "raw"
