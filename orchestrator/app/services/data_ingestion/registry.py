from typing import Dict, List, Optional
from .base import BaseIngestor, DataSource, IngestionJob
import logging

logger = logging.getLogger(__name__)

class IngestionRegistry:
    """Registry for managing different data ingestors"""
    
    def __init__(self):
        self._ingestors: Dict[str, BaseIngestor] = {}
    
    def register(self, ingestor: BaseIngestor):
        """Register a new ingestor"""
        for source_type in ingestor.supported_types:
            if source_type in self._ingestors:
                logger.warning(f"Overriding existing ingestor for type: {source_type}")
            self._ingestors[source_type] = ingestor
            logger.info(f"Registered ingestor {ingestor.name} for type: {source_type}")
    
    def get_ingestor(self, source_type: str) -> Optional[BaseIngestor]:
        """Get ingestor for a specific source type"""
        return self._ingestors.get(source_type)
    
    def list_supported_types(self) -> List[str]:
        """List all supported source types"""
        return list(self._ingestors.keys())
    
    def list_ingestors(self) -> Dict[str, BaseIngestor]:
        """Get all registered ingestors"""
        return self._ingestors.copy()

