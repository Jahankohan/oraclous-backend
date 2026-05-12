# Knowledge Graph Builder - Development Rules & Architecture Guidelines

## 🏗️ **CRITICAL ARCHITECTURE PRINCIPLE**

### **❌ NEVER CREATE DUPLICATE SERVICES**
### **✅ ALWAYS ENHANCE EXISTING SERVICES IN PLACE**

---

## 📋 **Core Development Rules**

### **Rule #1: One Service Per Functionality**
```bash
❌ WRONG: chat_service.py + enhanced_chat_service.py + advanced_chat_service.py
✅ CORRECT: chat_service.py (enhanced in place)

❌ WRONG: entity_extractor.py + advanced_entity_extractor.py
✅ CORRECT: entity_extractor.py (enhanced in place)

❌ WRONG: schema_service.py + dynamic_schema_service.py
✅ CORRECT: schema_service.py (enhanced in place)
```

### **Rule #2: Enhancement Pattern**
When improving functionality:
1. **Add new methods** to existing service
2. **Refactor existing methods** with backward compatibility
3. **Use feature flags** for gradual rollout if needed
4. **Update existing endpoints** to use enhanced methods
5. **Remove deprecated code** after migration

### **Rule #3: Endpoint Consistency**
```python
# ✅ CORRECT: One endpoint with enhanced functionality
@router.post("/graphs/{graph_id}/chat")
async def chat_with_graph(request: ChatRequest):
    # Enhanced implementation with backward compatibility
    return await chat_service.chat_with_graph(request)

# ❌ WRONG: Multiple endpoints for same functionality
@router.post("/graphs/{graph_id}/chat")           # Old
@router.post("/graphs/{graph_id}/chat/advanced")  # New
@router.post("/graphs/{graph_id}/chat/enhanced")  # Newer
```

### **Rule #4: Backward Compatibility**
Always maintain backward compatibility during enhancements:
```python
# ✅ CORRECT: Enhanced method with backward compatibility
async def chat_with_graph(
    self,
    query: str,
    mode: str = "comprehensive",  # New parameter with default
    reasoning_depth: int = 3,     # New parameter with default
    # ... existing parameters remain the same
) -> Dict[str, Any]:
    # Enhanced implementation that works with old and new calls
```

---

## 🔧 **Implementation Guidelines**

### **Enhancement Process:**
1. **Analyze Current Service** - Understand existing functionality
2. **Design Enhancement** - Plan new features within existing structure
3. **Refactor In Place** - Add new methods, enhance existing ones
4. **Update Tests** - Ensure all functionality still works
5. **Update Endpoints** - Connect enhanced service to endpoints
6. **Validate Integration** - Test end-to-end functionality

### **File Naming Convention:**
```bash
✅ KEEP: chat_service.py (enhanced)
✅ KEEP: entity_extractor.py (enhanced)
✅ KEEP: schema_service.py (enhanced)

❌ DELETE: enhanced_chat_service.py
❌ DELETE: advanced_graph_context.py (merge into chat_service.py)
❌ DELETE: any _v2, _advanced, _enhanced services
```

### **Method Naming Pattern:**
```python
# ✅ CORRECT: Clear, descriptive method names in same service
class ChatService:
    async def chat_with_graph(self, query: str, mode: str = "comprehensive"):
        """Main chat method with enhanced graph reasoning"""

    async def _generate_graph_context(self, query: str):
        """Private method for advanced context generation"""

    async def _validate_response_grounding(self, response: str):
        """Private method for hallucination prevention"""

# ❌ WRONG: Separate service for same functionality
class AdvancedGraphContextService:  # Should be part of ChatService
class EnhancedChatService:          # Should be ChatService enhancement
```

---

## 📁 **Current Architecture Issues TO FIX**

### **Issue #1: Duplicate Chat Services**
```bash
❌ CURRENT STATE:
- app/services/chat_service.py (original)
- app/services/enhanced_chat_service.py (duplicate)
- app/services/advanced_graph_context.py (should be part of chat_service)

✅ REQUIRED STATE:
- app/services/chat_service.py (enhanced with all functionality)
```

### **Issue #2: Broken Endpoints**
```bash
❌ CURRENT STATE: Many endpoints in graphs.py have no implementation
✅ REQUIRED STATE: All endpoints connect to working service methods
```

### **Issue #3: Service Dependencies**
```bash
❌ CURRENT STATE: Complex dependency chains between services
✅ REQUIRED STATE: Clear, simple dependencies with single responsibility
```

---

## 🛠️ **Refactoring Checklist**

### **For Chat Service:**
- [ ] Merge `advanced_graph_context.py` functionality into `chat_service.py`
- [ ] Merge `enhanced_chat_service.py` functionality into `chat_service.py`
- [ ] Update `chat.py` endpoints to use single enhanced service
- [ ] Remove duplicate service files
- [ ] Test all chat functionality works

### **For Entity Extractor:**
- [ ] Ensure all schema evolution code is in `entity_extractor.py`
- [ ] Remove any duplicate extractor services
- [ ] Update ingestion endpoints to use single service
- [ ] Test extraction pipeline works end-to-end

### **For All Services:**
- [ ] One service file per major functionality
- [ ] All enhancements done in place
- [ ] Endpoints connect to actual implementations
- [ ] No orphaned or unused service files
- [ ] Clear method organization within services

---

## 📋 **Service Responsibility Matrix**

| Service | Responsibility | Should NOT Handle |
|---------|----------------|-------------------|
| `chat_service.py` | Graph chat, context generation, response grounding | Entity extraction, schema management |
| `entity_extractor.py` | Entity extraction, schema evolution, graph document creation | Chat, search, embeddings |
| `schema_service.py` | Schema management, validation, database schema operations | Entity extraction, chat |
| `search_service.py` | Vector search, similarity, graph querying | Chat responses, entity extraction |
| `vector_service.py` | Vector operations, embeddings, Neo4j vector indexes | Chat, entity extraction |
| `graph_service.py` | Neo4j operations, graph storage, data management | Chat, schema learning |

---

## 🚨 **Code Review Checklist**

### **Before Any PR:**
- [ ] ❌ Did I create a new service instead of enhancing existing?
- [ ] ❌ Did I create duplicate endpoints?
- [ ] ❌ Did I break backward compatibility?
- [ ] ✅ Did I enhance existing service in place?
- [ ] ✅ Did I maintain single responsibility per service?
- [ ] ✅ Did I update relevant endpoints to use enhanced functionality?
- [ ] ✅ Did I test that existing functionality still works?

### **Service Enhancement Pattern:**
```python
# ✅ CORRECT Enhancement Pattern
class ChatService:
    def __init__(self):
        self.current_graph_id = None
        # ... existing attributes ...

        # NEW: Enhanced capabilities
        self.reasoning_modes = ["comprehensive", "focused", "exploratory"]
        self.graph_algorithms = GraphAlgorithms()  # New component

    # EXISTING method - enhanced with new parameters
    async def chat_with_graph(
        self,
        query: str,
        mode: str = "comprehensive",  # NEW parameter
        graph_id: Optional[UUID] = None,  # EXISTING parameter
        # ... other existing parameters
    ) -> Dict[str, Any]:
        # Enhanced implementation that includes new graph reasoning
        # while maintaining compatibility with existing calls

    # NEW method - added functionality
    async def explain_reasoning(self, query: str) -> Dict[str, Any]:
        """NEW: Explain reasoning process for query"""

    # EXISTING method - refactored for better performance
    async def _generate_response(self, query: str, context: str) -> str:
        # Refactored implementation with improved grounding validation
```

---

## 🎯 **Success Criteria**

### **Architecture Quality:**
- ✅ One service file per major functionality
- ✅ No duplicate services or endpoints
- ✅ Clear separation of concerns
- ✅ All endpoints have working implementations

### **Code Quality:**
- ✅ Backward compatibility maintained
- ✅ Clean method organization
- ✅ No orphaned code
- ✅ Comprehensive test coverage

### **Maintainability:**
- ✅ Easy to understand which service handles what
- ✅ Simple to add new features without creating new services
- ✅ Clear enhancement path for future improvements
- ✅ Minimal technical debt

---

## 🏆 **Golden Rule**

### **"ENHANCE, DON'T DUPLICATE"**

**Before creating any new service, ask:**
1. **Can this be added to an existing service?** (Usually YES)
2. **Does this break single responsibility principle?** (If YES, refactor existing service)
3. **Am I solving the right problem?** (Enhancement vs new service)

**The answer is almost always to enhance the existing service, not create a new one.**

---

## 📞 **When to Create New Service (Rare Cases)**

**✅ Create new service ONLY when:**
- Completely different domain (e.g., adding user management to graph builder)
- External integration (e.g., new payment provider)
- Independent microservice (e.g., separate notification service)

**❌ DON'T create new service for:**
- Enhanced functionality of existing service
- "Advanced" or "improved" versions
- Different algorithms for same purpose
- Feature flags or modes

---

**This document should be the foundation for all future development work on this project.**
