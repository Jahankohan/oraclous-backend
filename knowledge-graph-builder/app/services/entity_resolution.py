import logging
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
import re
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.core.neo4j_client import Neo4jClient
from app.core.exceptions import ServiceError
from app.config.settings import get_settings
from app.utils.llm_clients import LLMClientFactory

logger = logging.getLogger(__name__)

class EntityResolution:
    """Advanced entity resolution and deduplication"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.settings = get_settings()
        self.llm_factory = LLMClientFactory()
        
        # Similarity thresholds for different matching strategies
        self.thresholds = {
            "exact_match": 1.0,
            "normalized_match": 0.95,
            "fuzzy_match": 0.85,
            "embedding_match": 0.90,
            "composite_match": 0.80
        }
    
    async def find_duplicate_entities(self, batch_size: int = 1000) -> List[Dict[str, Any]]:
        """Find duplicate entities using multiple strategies"""
        try:
            # Get all entities in batches
            all_duplicates = []
            offset = 0
            
            while True:
                query = """
                MATCH (e:Entity)
                WHERE e.name IS NOT NULL
                RETURN e.id as id, e.name as name, e.embedding as embedding, 
                       labels(e) as labels, properties(e) as properties
                SKIP $offset LIMIT $limit
                """
                
                batch_entities = self.neo4j.execute_query(query, {
                    "offset": offset,
                    "limit": batch_size
                })
                
                if not batch_entities:
                    break
                
                # Find duplicates within this batch
                batch_duplicates = await self._find_duplicates_in_batch(batch_entities)
                all_duplicates.extend(batch_duplicates)
                
                offset += batch_size
            
            # Merge duplicate groups and remove redundancies
            merged_duplicates = self._merge_duplicate_groups(all_duplicates)
            
            # Score and rank duplicate groups
            scored_duplicates = await self._score_duplicate_groups(merged_duplicates)
            
            return sorted(scored_duplicates, key=lambda x: x["confidence_score"], reverse=True)
            
        except Exception as e:
            logger.error(f"Duplicate entity detection failed: {e}")
            raise ServiceError(f"Duplicate entity detection failed: {e}")
    
    async def resolve_entity_duplicates(self, duplicate_groups: List[Dict[str, Any]], auto_merge: bool = False) -> Dict[str, Any]:
        """Resolve duplicate entities by merging them"""
        results = {
            "merged_groups": 0,
            "merged_entities": 0,
            "skipped_groups": 0,
            "errors": []
        }
        
        for group in duplicate_groups:
            try:
                if auto_merge and group["confidence_score"] > 0.95:
                    # Auto-merge high confidence duplicates
                    canonical_entity = self._select_canonical_entity(group["entities"])
                    await self._merge_entity_group(group["entities"], canonical_entity["id"])
                    results["merged_groups"] += 1
                    results["merged_entities"] += len(group["entities"]) - 1
                elif not auto_merge:
                    # Manual review needed - store suggestions
                    await self._store_merge_suggestion(group)
                    results["skipped_groups"] += 1
                
            except Exception as e:
                error_msg = f"Failed to process group {group.get('group_id', 'unknown')}: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
        
        return results
    
    async def _find_duplicates_in_batch(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find duplicates within a batch of entities"""
        duplicates = []
        processed_pairs = set()
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], start=i+1):
                pair_key = tuple(sorted([entity1["id"], entity2["id"]]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                # Calculate similarity using multiple strategies
                similarity_scores = await self._calculate_entity_similarity(entity1, entity2)
                
                # Check if any similarity exceeds threshold
                if any(score >= self.thresholds.get(method, 0.8) 
                      for method, score in similarity_scores.items()):
                    
                    duplicates.append({
                        "entities": [entity1, entity2],
                        "similarity_scores": similarity_scores,
                        "primary_method": max(similarity_scores.items(), key=lambda x: x[1])[0]
                    })
        
        return duplicates
    
    async def _calculate_entity_similarity(self, entity1: Dict, entity2: Dict) -> Dict[str, float]:
        """Calculate similarity between two entities using multiple methods"""
        similarities = {}
        
        name1, name2 = entity1["name"], entity2["name"]
        
        # 1. Exact match
        similarities["exact_match"] = 1.0 if name1 == name2 else 0.0
        
        # 2. Normalized match (case, whitespace, punctuation insensitive)
        norm_name1 = self._normalize_entity_name(name1)
        norm_name2 = self._normalize_entity_name(name2)
        similarities["normalized_match"] = 1.0 if norm_name1 == norm_name2 else 0.0
        
        # 3. Fuzzy string matching
        similarities["fuzzy_match"] = SequenceMatcher(None, norm_name1, norm_name2).ratio()
        
        # 4. Jaccard similarity of words
        words1 = set(norm_name1.split())
        words2 = set(norm_name2.split())
        if words1 or words2:
            similarities["jaccard_words"] = len(words1 & words2) / len(words1 | words2)
        else:
            similarities["jaccard_words"] = 0.0
        
        # 5. Embedding similarity (if available)
        emb1, emb2 = entity1.get("embedding"), entity2.get("embedding")
        if emb1 and emb2:
            similarities["embedding_match"] = float(cosine_similarity([emb1], [emb2])[0][0])
        
        # 6. Label compatibility
        labels1 = set(entity1.get("labels", []))
        labels2 = set(entity2.get("labels", []))
        if labels1 or labels2:
            similarities["label_compatibility"] = len(labels1 & labels2) / len(labels1 | labels2)
        else:
            similarities["label_compatibility"] = 1.0  # Both have no specific labels
        
        # 7. Property similarity
        props1 = entity1.get("properties", {})
        props2 = entity2.get("properties", {})
        similarities["property_similarity"] = self._calculate_property_similarity(props1, props2)
        
        # 8. Composite score (weighted average)
        weights = {
            "exact_match": 0.3,
            "normalized_match": 0.25,
            "fuzzy_match": 0.2,
            "embedding_match": 0.15,
            "jaccard_words": 0.05,
            "label_compatibility": 0.03,
            "property_similarity": 0.02
        }
        
        composite_score = 0.0
        total_weight = 0.0
        for method, score in similarities.items():
            if score is not None and method in weights:
                composite_score += score * weights[method]
                total_weight += weights[method]
        
        if total_weight > 0:
            similarities["composite_match"] = composite_score / total_weight
        
        return similarities
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for comparison"""
        if not name:
            return ""
        
        # Convert to lowercase
        normalized = name.lower()
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        # Remove common punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Handle common abbreviations and variations
        abbreviations = {
            r'\binc\b': 'incorporated',
            r'\bcorp\b': 'corporation',
            r'\bco\b': 'company',
            r'\bllc\b': 'limited liability company',
            r'\bltd\b': 'limited',
            r'\buniv\b': 'university',
            r'\bdept\b': 'department',
            r'\bdr\b': 'doctor',
            r'\bprof\b': 'professor'
        }
        
        for abbrev, full_form in abbreviations.items():
            normalized = re.sub(abbrev, full_form, normalized)
        
        return normalized.strip()
    
    def _calculate_property_similarity(self, props1: Dict, props2: Dict) -> float:
        """Calculate similarity between entity properties"""
        if not props1 and not props2:
            return 1.0
        
        if not props1 or not props2:
            return 0.0
        
        # Remove system properties
        system_props = {'id', 'name', 'embedding', 'community', 'degree'}
        filtered_props1 = {k: v for k, v in props1.items() if k not in system_props}
        filtered_props2 = {k: v for k, v in props2.items() if k not in system_props}
        
        if not filtered_props1 and not filtered_props2:
            return 1.0
        
        # Calculate Jaccard similarity of property keys
        keys1, keys2 = set(filtered_props1.keys()), set(filtered_props2.keys())
        key_similarity = len(keys1 & keys2) / len(keys1 | keys2) if (keys1 | keys2) else 1.0
        
        # Calculate value similarity for common keys
        common_keys = keys1 & keys2
        value_similarity = 0.0
        
        if common_keys:
            for key in common_keys:
                val1, val2 = str(filtered_props1[key]), str(filtered_props2[key])
                val_sim = SequenceMatcher(None, val1.lower(), val2.lower()).ratio()
                value_similarity += val_sim
            value_similarity /= len(common_keys)
        
        return (key_similarity + value_similarity) / 2
    
    def _merge_duplicate_groups(self, duplicates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge overlapping duplicate groups"""
        # Build graph of duplicate relationships
        G = nx.Graph()
        
        for dup in duplicates:
            entity_ids = [e["id"] for e in dup["entities"]]
            if len(entity_ids) == 2:
                G.add_edge(entity_ids[0], entity_ids[1], 
                          similarity_scores=dup["similarity_scores"],
                          primary_method=dup["primary_method"])
        
        # Find connected components (groups of related duplicates)
        merged_groups = []
        for component in nx.connected_components(G):
            if len(component) >= 2:
                # Get entities for this group
                entities_dict = {}
                for dup in duplicates:
                    for entity in dup["entities"]:
                        if entity["id"] in component:
                            entities_dict[entity["id"]] = entity
                
                group_entities = list(entities_dict.values())
                
                # Calculate average similarity within group
                edges_in_component = G.subgraph(component).edges(data=True)
                avg_similarity = 0.0
                if edges_in_component:
                    similarities = [edge_data["similarity_scores"].get("composite_match", 0.0)
                                  for _, _, edge_data in edges_in_component]
                    avg_similarity = np.mean(similarities)
                
                merged_groups.append({
                    "group_id": f"group_{len(merged_groups)}",
                    "entities": group_entities,
                    "size": len(group_entities),
                    "avg_similarity": avg_similarity
                })
        
        return merged_groups
    
    async def _score_duplicate_groups(self, groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score duplicate groups for confidence"""
        for group in groups:
            # Base score from similarity
            base_score = group["avg_similarity"]
            
            # Adjust based on group size (larger groups are less reliable)
            size_penalty = max(0.0, (group["size"] - 2) * 0.1)
            
            # Adjust based on entity name lengths (very short names are less reliable)
            name_lengths = [len(e["name"]) for e in group["entities"]]
            avg_name_length = np.mean(name_lengths)
            length_bonus = min(0.1, avg_name_length / 50)  # Bonus for longer names
            
            # Adjust based on label consistency
            all_labels = [set(e.get("labels", [])) for e in group["entities"]]
            if all_labels:
                label_intersection = set.intersection(*all_labels)
                label_union = set.union(*all_labels)
                label_consistency = len(label_intersection) / len(label_union) if label_union else 1.0
            else:
                label_consistency = 1.0
            
            # Calculate final confidence score
            confidence = base_score - size_penalty + length_bonus + (label_consistency * 0.05)
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            
            group["confidence_score"] = confidence
            group["scoring_details"] = {
                "base_score": base_score,
                "size_penalty": size_penalty,
                "length_bonus": length_bonus,
                "label_consistency": label_consistency
            }
        
        return groups
    
    def _select_canonical_entity(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the canonical entity from a duplicate group"""
        # Score each entity for canonicality
        scores = []
        
        for entity in entities:
            score = 0.0
            
            # Prefer longer names (more descriptive)
            score += len(entity["name"]) * 0.1
            
            # Prefer entities with more properties
            props = entity.get("properties", {})
            score += len(props) * 2
            
            # Prefer entities with embeddings
            if entity.get("embedding"):
                score += 10
            
            # Prefer entities with more specific labels
            labels = entity.get("labels", [])
            specific_labels = [l for l in labels if l != "Entity"]
            score += len(specific_labels) * 3
            
            scores.append((entity, score))
        
        # Return entity with highest score
        return max(scores, key=lambda x: x[1])[0]
    
    async def _merge_entity_group(self, entities: List[Dict[str, Any]], canonical_id: str) -> None:
        """Merge a group of duplicate entities into the canonical entity"""
        other_ids = [e["id"] for e in entities if e["id"] != canonical_id]
        
        if not other_ids:
            return
        
        # Merge properties and relationships
        merge_query = """
        MATCH (canonical:Entity {id: $canonicalId})
        MATCH (duplicate:Entity) WHERE duplicate.id IN $duplicateIds
        
        // Merge properties (keep canonical, add non-conflicting from duplicates)
        WITH canonical, collect(duplicate) as duplicates
        UNWIND duplicates as dup
        WITH canonical, dup, properties(dup) as dupProps
        UNWIND keys(dupProps) as key
        WITH canonical, dup, key, dupProps[key] as value
        WHERE NOT key IN keys(properties(canonical)) OR properties(canonical)[key] IS NULL
        SET canonical += {[key]: value}
        
        // Redirect all relationships from duplicates to canonical
        WITH canonical, collect(DISTINCT dup) as allDuplicates
        UNWIND allDuplicates as duplicate
        MATCH (duplicate)-[r]-(other)
        WHERE other <> canonical AND NOT (canonical)-[r]-(other)
        CREATE (canonical)-[newR:TYPE(r)]->(other)
        SET newR += properties(r)
        
        // Delete duplicate entities and their relationships
        DETACH DELETE duplicate
        
        RETURN count(allDuplicates) as mergedCount
        """
        
        # Note: The above query has some Cypher syntax that might need adjustment
        # Here's a more conservative approach:
        
        for duplicate_id in other_ids:
            # Get duplicate entity relationships
            rel_query = """
            MATCH (duplicate:Entity {id: $duplicateId})-[r]-(other)
            RETURN type(r) as relType, other.id as otherId, 
                   properties(r) as relProps, startNode(r) = duplicate as outgoing
            """
            
            relationships = self.neo4j.execute_query(rel_query, {"duplicateId": duplicate_id})
            
            # Create new relationships from canonical entity
            for rel in relationships:
                if rel["outgoing"]:
                    create_rel_query = f"""
                    MATCH (canonical:Entity {{id: $canonicalId}})
                    MATCH (other:Entity {{id: $otherId}})
                    WHERE NOT (canonical)-[:{rel["relType"]}]->(other)
                    CREATE (canonical)-[r:{rel["relType"]}]->(other)
                    SET r += $relProps
                    """
                else:
                    create_rel_query = f"""
                    MATCH (canonical:Entity {{id: $canonicalId}})
                    MATCH (other:Entity {{id: $otherId}})
                    WHERE NOT (other)-[:{rel["relType"]}]->(canonical)
                    CREATE (other)-[r:{rel["relType"]}]->(canonical)
                    SET r += $relProps
                    """
                
                try:
                    self.neo4j.execute_write_query(create_rel_query, {
                        "canonicalId": canonical_id,
                        "otherId": rel["otherId"],
                        "relProps": rel["relProps"]
                    })
                except Exception as e:
                    logger.warning(f"Failed to create relationship: {e}")
            
            # Delete duplicate entity
            delete_query = """
            MATCH (duplicate:Entity {id: $duplicateId})
            DETACH DELETE duplicate
            """
            
            self.neo4j.execute_write_query(delete_query, {"duplicateId": duplicate_id})
        
        logger.info(f"Merged {len(other_ids)} entities into {canonical_id}")
    
    async def _store_merge_suggestion(self, group: Dict[str, Any]) -> None:
        """Store merge suggestion for manual review"""
        suggestion_query = """
        CREATE (suggestion:MergeSuggestion {
            id: $groupId,
            entityIds: $entityIds,
            confidenceScore: $confidenceScore,
            createdAt: datetime(),
            status: 'pending'
        })
        """
        
        self.neo4j.execute_write_query(suggestion_query, {
            "groupId": group["group_id"],
            "entityIds": [e["id"] for e in group["entities"]],
            "confidenceScore": group["confidence_score"]
        })


class SchemaLearning:
    """Advanced schema learning and evolution"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.settings = get_settings()
        self.llm_factory = LLMClientFactory()
    
    async def learn_schema_from_data(self, domain_context: Optional[str] = None) -> Dict[str, Any]:
        """Learn schema patterns from existing data"""
        try:
            # Analyze existing entities and relationships
            current_schema = await self._analyze_current_schema()
            
            # Identify patterns and suggest improvements
            suggestions = await self._generate_schema_suggestions(current_schema, domain_context)
            
            # Validate suggestions against data
            validated_suggestions = await self._validate_schema_suggestions(suggestions)
            
            return {
                "current_schema": current_schema,
                "suggestions": validated_suggestions,
                "improvement_areas": self._identify_improvement_areas(current_schema)
            }
            
        except Exception as e:
            logger.error(f"Schema learning failed: {e}")
            raise ServiceError(f"Schema learning failed: {e}")
    
    async def _analyze_current_schema(self) -> Dict[str, Any]:
        """Analyze current schema patterns"""
        schema_analysis = {}
        
        # 1. Entity label distribution
        label_query = """
        MATCH (e:Entity)
        RETURN labels(e) as labels, count(e) as count
        ORDER BY count DESC
        """
        
        label_stats = self.neo4j.execute_query(label_query)
        schema_analysis["entity_labels"] = label_stats
        
        # 2. Relationship type distribution
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) as relationshipType, count(r) as count
        ORDER BY count DESC
        """
        
        rel_stats = self.neo4j.execute_query(rel_query)
        schema_analysis["relationship_types"] = rel_stats
        
        # 3. Common patterns (label combinations)
        pattern_query = """
        MATCH (e1:Entity)-[r]->(e2:Entity)
        RETURN labels(e1) as sourceLabels, type(r) as relType, 
               labels(e2) as targetLabels, count(*) as frequency
        ORDER BY frequency DESC
        LIMIT 50
        """
        
        patterns = self.neo4j.execute_query(pattern_query)
        schema_analysis["common_patterns"] = patterns
        
        # 4. Property usage
        prop_query = """
        MATCH (e:Entity)
        UNWIND keys(properties(e)) as propKey
        RETURN propKey, count(e) as entityCount
        ORDER BY entityCount DESC
        """
        
        properties = self.neo4j.execute_query(prop_query)
        schema_analysis["property_usage"] = properties
        
        return schema_analysis
    
    async def _generate_schema_suggestions(self, current_schema: Dict, domain_context: Optional[str]) -> List[Dict[str, Any]]:
        """Generate schema improvement suggestions using LLM"""
        try:
            llm = self.llm_factory.get_llm(self.settings.default_llm_model)
            
            # Prepare context for LLM
            context = f"""
            Current Knowledge Graph Schema Analysis:
            
            Entity Labels: {current_schema.get('entity_labels', [])}
            Relationship Types: {current_schema.get('relationship_types', [])}
            Common Patterns: {current_schema.get('common_patterns', [])}
            Property Usage: {current_schema.get('property_usage', [])}
            
            Domain Context: {domain_context or 'General domain'}
            
            Please analyze this schema and suggest improvements in the following areas:
            1. Missing entity types that would be valuable
            2. Missing relationship types that would provide better connectivity
            3. Redundant or overly generic labels that could be more specific
            4. Property standardization opportunities
            5. Hierarchical relationships that could improve organization
            
            Provide your response in JSON format with the following structure:
            {{
                "entity_suggestions": [
                    {{"type": "add", "label": "NewLabel", "reason": "Why this would be valuable"}},
                    {{"type": "rename", "from": "OldLabel", "to": "NewLabel", "reason": "Why this is better"}}
                ],
                "relationship_suggestions": [...],
                "property_suggestions": [...],
                "hierarchy_suggestions": [...]
            }}
            """
            
            response = await asyncio.to_thread(llm.predict, context)
            
            # Parse LLM response (would need more robust JSON parsing)
            import json
            try:
                suggestions = json.loads(response)
            except json.JSONDecodeError:
                # Fallback to rule-based suggestions if LLM response is not valid JSON
                suggestions = self._generate_rule_based_suggestions(current_schema)
            
            return suggestions
            
        except Exception as e:
            logger.warning(f"LLM schema suggestion failed, using rule-based: {e}")
            return self._generate_rule_based_suggestions(current_schema)
    
    def _generate_rule_based_suggestions(self, current_schema: Dict) -> Dict[str, Any]:
        """Generate schema suggestions using rule-based approach"""
        suggestions = {
            "entity_suggestions": [],
            "relationship_suggestions": [],
            "property_suggestions": [],
            "hierarchy_suggestions": []
        }
        
        # Analyze entity labels for potential improvements
        entity_labels = current_schema.get("entity_labels", [])
        
        # Suggest more specific labels for generic ones
        generic_labels = ["Entity", "Thing", "Object", "Item"]
        for label_info in entity_labels:
            labels = label_info.get("labels", [])
            if any(generic in labels for generic in generic_labels):
                suggestions["entity_suggestions"].append({
                    "type": "specify",
                    "current_labels": labels,
                    "reason": "Generic labels could be made more specific"
                })
        
        # Suggest common relationship types if missing
        rel_types = {r["relationshipType"] for r in current_schema.get("relationship_types", [])}
        common_relations = ["PART_OF", "LOCATED_IN", "RELATED_TO", "BELONGS_TO", "CONTAINS"]
        
        for common_rel in common_relations:
            if common_rel not in rel_types:
                suggestions["relationship_suggestions"].append({
                    "type": "add",
                    "relationship": common_rel,
                    "reason": f"Common relationship type {common_rel} not found in schema"
                })
        
        return suggestions
    
    async def _validate_schema_suggestions(self, suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """Validate schema suggestions against actual data"""
        validated = suggestions.copy()
        
        # Add validation scores and feasibility assessments
        for suggestion_type in ["entity_suggestions", "relationship_suggestions"]:
            if suggestion_type in validated:
                for suggestion in validated[suggestion_type]:
                    # Add feasibility score based on data analysis
                    suggestion["feasibility_score"] = await self._calculate_suggestion_feasibility(suggestion)
                    suggestion["impact_estimate"] = await self._estimate_suggestion_impact(suggestion)
        
        return validated
    
    async def _calculate_suggestion_feasibility(self, suggestion: Dict) -> float:
        """Calculate how feasible a schema suggestion is"""
        # Simple heuristic-based feasibility scoring
        suggestion_type = suggestion.get("type", "")
        
        if suggestion_type == "add":
            return 0.8  # Adding new elements is usually feasible
        elif suggestion_type == "rename":
            return 0.6  # Renaming requires data migration
        elif suggestion_type == "specify":
            return 0.4  # Specifying generic labels requires careful analysis
        
        return 0.5  # Default
    
    async def _estimate_suggestion_impact(self, suggestion: Dict) -> str:
        """Estimate the impact of implementing a suggestion"""
        suggestion_type = suggestion.get("type", "")
        
        if suggestion_type == "add":
            return "medium"  # New elements add capability
        elif suggestion_type == "rename":
            return "high"   # Renaming affects existing queries
        elif suggestion_type == "specify":
            return "high"   # More specific schema improves precision
        
        return "low"
    
    def _identify_improvement_areas(self, current_schema: Dict) -> List[Dict[str, Any]]:
        """Identify specific areas where the schema could be improved"""
        improvements = []
        
        # Check for schema consistency issues
        entity_labels = current_schema.get("entity_labels", [])
        rel_types = current_schema.get("relationship_types", [])
        
        # 1. Label diversity (too many singleton labels)
        singleton_labels = [label for label in entity_labels if label.get("count", 0) == 1]
        if len(singleton_labels) > 10:
            improvements.append({
                "area": "label_consolidation",
                "issue": f"Many singleton entity labels ({len(singleton_labels)})",
                "suggestion": "Consider consolidating rare labels into more general categories"
            })
        
        # 2. Generic relationship overuse
        generic_rels = [r for r in rel_types if r.get("relationshipType") in ["RELATED_TO", "CONNECTED_TO"]]
        total_rels = sum(r.get("count", 0) for r in rel_types)
        generic_percentage = sum(r.get("count", 0) for r in generic_rels) / max(total_rels, 1)
        
        if generic_percentage > 0.5:
            improvements.append({
                "area": "relationship_specificity",
                "issue": f"High percentage ({generic_percentage:.1%}) of generic relationships",
                "suggestion": "Replace generic relationships with more specific types"
            })
        
        return improvements