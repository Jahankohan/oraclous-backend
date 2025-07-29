from app.models.base_model import Base
from sqlalchemy import Column, String, Text, ARRAY, TIMESTAMP, MetaData
from sqlalchemy import ForeignKey

class OauthTokens(Base):
    __tablename__ = "oauth_tokens"

    user_id = Column(String, ForeignKey("users.id"), primary_key=True)
    provider = Column(String, primary_key=True)
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text)
    scopes = Column(ARRAY(String))
    expires_at = Column(TIMESTAMP)
    auth_metadata = MetaData()
