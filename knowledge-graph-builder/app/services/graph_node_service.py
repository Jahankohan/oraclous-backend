"""
Neo4j Graph Node Service

This service manages Graph nodes in Neo4j, replacing PostgreSQL knowledge_graphs table.
Each Graph node represents a knowledge graph with metadata and tenant isolation.

Graph Node Schema:
(graph:Graph {
    graph_id: "uuid",
    name: "Graph Name", 
    description: "Optional description",
    user_id: "user_uuid",
    created_at: datetime(),
    updated_at: datetime(),
    node_count: 0,
    relationship_count: 0,
    status: "active"
})
"""

from typing import Optional, Dict, Any, List
from neo4j import Driver
from neo4j.exceptions import Neo4jError


class GraphNodeService:
    """Service for managing Graph nodes in Neo4j"""
    
    def __init__(self, driver: Driver):
        """
        Initialize GraphNodeService with Neo4j driver
        
        Args:
            driver: Neo4j driver instance
        """
        self.driver = driver
    
    def create_graph(
        self, 
        graph_id: str, 
        name: str, 
        user_id: str, 
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new Graph node in Neo4j"""
        
        query = """
        CREATE (g:Graph {
            graph_id: $graph_id,
            name: $name,
            description: $description,
            user_id: $user_id,
            created_at: datetime(),
            updated_at: datetime(),
            node_count: 0,
            relationship_count: 0,
            status: 'active'
        })
        RETURN g
        """
        
        try:
            if not self.driver:
                raise ValueError("Neo4j driver not initialized")
                
            with self.driver.session() as session:
                result = session.run(query, {
                    'graph_id': graph_id,
                    'name': name,
                    'description': description or '',
                    'user_id': user_id
                })
                
                record = result.single()
                if not record:
                    raise ValueError("Failed to create Graph node")
                    
                graph_data = dict(record["g"])
                return {
                    'graph_id': graph_data['graph_id'],
                    'name': graph_data['name'],
                    'description': graph_data['description'],
                    'user_id': graph_data['user_id'],
                    'created_at': graph_data['created_at'].isoformat(),
                    'updated_at': graph_data['updated_at'].isoformat(),
                    'node_count': graph_data['node_count'],
                    'relationship_count': graph_data['relationship_count'],
                    'status': graph_data['status']
                }
                
        except Neo4jError as e:
            raise Exception(f"Failed to create graph: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error creating graph: {str(e)}")
    
    def get_graph(self, graph_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a Graph node by graph_id.
        
        Args:
            graph_id: Graph identifier
            
        Returns:
            Graph metadata dict or None if not found
        """
        query = """
        MATCH (g:Graph {graph_id: $graph_id})
        RETURN g {
            .graph_id,
            .name,
            .description,
            .user_id,
            .created_at,
            .updated_at,
            .node_count,
            .relationship_count,
            .status
        } as graph
        """
        
        try:
            if not self.driver:
                raise ValueError("Neo4j driver not initialized")
                
            with self.driver.session() as session:
                result = session.run(query, {'graph_id': graph_id})
                record = result.single()
                
                if record:
                    return dict(record['graph'])
                return None
                
        except Neo4jError as e:
            raise Exception(f"Failed to retrieve Graph node {graph_id}: {e}")
    
    def list_user_graphs(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all Graph nodes for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of graph metadata dicts
        """
        query = """
        MATCH (g:Graph {user_id: $user_id})
        RETURN g {
            .graph_id,
            .name,
            .description,
            .user_id,
            .created_at,
            .updated_at,
            .node_count,
            .relationship_count,
            .status
        } as graph
        ORDER BY g.created_at DESC
        """
        
        try:
            if not self.driver:
                raise ValueError("Neo4j driver not initialized")
                
            with self.driver.session() as session:
                result = session.run(query, {'user_id': user_id})
                
                graphs: List[Dict[str, Any]] = []
                for record in result:
                    graphs.append(dict(record['graph']))
                
                return graphs
                
        except Neo4jError as e:
            raise Exception(f"Failed to list user graphs: {e}")
    
    def update_graph(
        self, 
        graph_id: str, 
        user_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        node_count: Optional[int] = None,
        relationship_count: Optional[int] = None,
        status: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update Graph node metadata.
        
        Args:
            graph_id: Graph identifier
            user_id: User identifier (for tenant isolation)
            name: New graph name
            description: New description
            node_count: Updated node count
            relationship_count: Updated relationship count
            status: New status
            
        Returns:
            Updated graph metadata dict or None if not found
        """
        # Build SET clause dynamically based on provided parameters
        set_parts = ["g.updated_at = datetime()"]
        params: Dict[str, Any] = {
            'graph_id': graph_id,
            'user_id': user_id
        }
        
        if name is not None:
            set_parts.append("g.name = $name")
            params['name'] = name
            
        if description is not None:
            set_parts.append("g.description = $description")
            params['description'] = description
            
        if node_count is not None:
            set_parts.append("g.node_count = $node_count")
            params['node_count'] = node_count
            
        if relationship_count is not None:
            set_parts.append("g.relationship_count = $relationship_count")
            params['relationship_count'] = relationship_count
            
        if status is not None:
            set_parts.append("g.status = $status")
            params['status'] = status
        
        # Use predefined queries for different update scenarios
        if len(set_parts) == 1:  # Only updating timestamp
            query = """
            MATCH (g:Graph {graph_id: $graph_id, user_id: $user_id})
            SET g.updated_at = datetime()
            RETURN g {
                .graph_id,
                .name,
                .description,
                .user_id,
                .created_at,
                .updated_at,
                .node_count,
                .relationship_count,
                .status
            } as graph
            """
        else:
            # For simplicity, handle most common update case
            query = """
            MATCH (g:Graph {graph_id: $graph_id, user_id: $user_id})
            SET g.updated_at = datetime()
            """
            
            if name is not None:
                query += ", g.name = $name"
            if description is not None:
                query += ", g.description = $description"
            if node_count is not None:
                query += ", g.node_count = $node_count"
            if relationship_count is not None:
                query += ", g.relationship_count = $relationship_count"
            if status is not None:
                query += ", g.status = $status"
                
            query += """
            RETURN g {
                .graph_id,
                .name,
                .description,
                .user_id,
                .created_at,
                .updated_at,
                .node_count,
                .relationship_count,
                .status
            } as graph
            """
        
        try:
            if not self.driver:
                raise ValueError("Neo4j driver not initialized")
                
            with self.driver.session() as session:
                result = session.run(query, params)
                record = result.single()
                
                if record:
                    return dict(record['graph'])
                return None
                
        except Neo4jError as e:
            raise Exception(f"Failed to update graph: {e}")
    
    def delete_graph(self, graph_id: str, user_id: str) -> bool:
        """
        Delete a Graph node and all its relationships.
        
        Args:
            graph_id: Graph identifier
            user_id: User identifier (for tenant isolation)
            
        Returns:
            True if deleted, False if not found
        """
        query = """
        MATCH (g:Graph {graph_id: $graph_id, user_id: $user_id})
        DETACH DELETE g
        RETURN count(g) as deleted_count
        """
        
        try:
            if not self.driver:
                raise ValueError("Neo4j driver not initialized")
                
            with self.driver.session() as session:
                result = session.run(query, {
                    'graph_id': graph_id,
                    'user_id': user_id
                })
                
                record = result.single()
                if record:
                    return record['deleted_count'] > 0
                return False
                
        except Neo4jError as e:
            raise Exception(f"Failed to delete graph: {e}")
    
    def graph_exists(self, graph_id: str, user_id: str) -> bool:
        """
        Check if a Graph node exists for the given user.
        
        Args:
            graph_id: Graph identifier
            user_id: User identifier
            
        Returns:
            True if graph exists, False otherwise
        """
        query = """
        MATCH (g:Graph {graph_id: $graph_id, user_id: $user_id})
        RETURN count(g) > 0 as exists
        """
        
        try:
            if not self.driver:
                raise ValueError("Neo4j driver not initialized")
                
            with self.driver.session() as session:
                result = session.run(query, {
                    'graph_id': graph_id,
                    'user_id': user_id
                })
                
                record = result.single()
                if record:
                    return record['exists']
                return False
                
        except Neo4jError as e:
            raise Exception(f"Failed to check graph existence: {e}")
