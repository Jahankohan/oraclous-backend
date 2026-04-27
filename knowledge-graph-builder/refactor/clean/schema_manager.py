"""Schema management - exact same functionality as original."""

from typing import Dict, Any
from neo4j_graphrag.experimental.components.schema import (
    SchemaFromTextExtractor,
    GraphSchema,
    NodeType,
    RelationshipType,
    PropertyType
)


class AdvancedSchemaManager:
    """Advanced schema management with learning and evolution - exact copy from original"""
    
    def __init__(self, config, llm):
        self.config = config
        self.llm = llm
        self.schema_extractor = SchemaFromTextExtractor(llm=llm) if config.enable_schema_learning else None 
        import logging
        self.logger = logging.getLogger(__name__)
        
        # Default enterprise schema
        self.default_schema = GraphSchema(
            node_types=[
                NodeType(
                    label="Person",
                    description="Individual human being with biographical information",
                    properties=[
                        PropertyType(name="name", type="STRING", required=True),
                        PropertyType(name="occupation", type="STRING"),
                        PropertyType(name="nationality", type="STRING"),
                        PropertyType(name="birth_date", type="DATE")
                    ]
                ),
                NodeType(
                    label="Organization",
                    description="Structured entity with business or institutional purpose",
                    properties=[
                        PropertyType(name="name", type="STRING", required=True),
                        PropertyType(name="industry", type="STRING"),
                        PropertyType(name="headquarters", type="STRING"),
                        PropertyType(name="founded_date", type="DATE")
                    ]
                ),
                NodeType(
                    label="Concept",
                    description="Abstract concept or topic",
                    properties=[
                        PropertyType(name="name", type="STRING", required=True),
                        PropertyType(name="description", type="STRING"),
                        PropertyType(name="domain", type="STRING")
                    ]
                )
            ],
            relationship_types=[
                RelationshipType(
                    label="WORKS_FOR",
                    description="Employment relationship",
                    properties=[
                        PropertyType(name="start_date", type="DATE"),
                        PropertyType(name="position", type="STRING")
                    ]
                ),
                RelationshipType(
                    label="RELATED_TO",
                    description="General relationship between entities",
                    properties=[
                        PropertyType(name="relationship_type", type="STRING"),
                        PropertyType(name="confidence", type="FLOAT")
                    ]
                )
            ],
            additional_node_types=not config.enforce_schema,
            additional_relationship_types=not config.enforce_schema
        ) 
    
    async def get_or_learn_schema(self, text_sample: str = None) -> GraphSchema:
        """Get existing schema or learn from text sample"""
        if self.config.enable_schema_learning and text_sample and self.schema_extractor:
            try:
                self.logger.info("Learning schema from text sample...")
                learned_schema = await self.schema_extractor.run(text=text_sample) 
                
                # Merge with default schema
                return self._merge_schemas(self.default_schema, learned_schema)
            except Exception as e:
                self.logger.warning(f"Schema learning failed, using default: {e}")
        
        return self.default_schema
    
    def _merge_schemas(self, base_schema: GraphSchema, learned_schema: GraphSchema) -> GraphSchema:
        """Merge learned schema with base schema"""
        # Simple merge - in production, this would be more sophisticated
        merged_nodes = base_schema.node_types + [
            node for node in learned_schema.node_types 
            if node.label not in [n.label for n in base_schema.node_types]
        ]
        
        merged_relationships = base_schema.relationship_types + [
            rel for rel in learned_schema.relationship_types
            if rel.label not in [r.label for r in base_schema.relationship_types]
        ]
        
        return GraphSchema(
            node_types=merged_nodes,
            relationship_types=merged_relationships,
            additional_node_types=base_schema.additional_node_types,
            additional_relationship_types=base_schema.additional_relationship_types
        )
