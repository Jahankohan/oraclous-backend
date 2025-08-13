class Neo4jConnectionError(Exception):
    """Exception raised when Neo4j connection fails"""
    pass

class ServiceError(Exception):
    """Base exception for service errors"""
    pass

class DocumentProcessingError(ServiceError):
    """Exception raised during document processing"""
    pass

class ExtractionError(ServiceError):
    """Exception raised during graph extraction"""
    pass

class EmbeddingError(ServiceError):
    """Exception raised during embedding generation"""
    pass

class ChatServiceError(ServiceError):
    """Exception raised in chat service"""
    pass
