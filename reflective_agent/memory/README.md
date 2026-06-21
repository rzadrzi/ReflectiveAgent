# Memory Management System

## Overview

The memory system is the core component that enables the agent's **self-improvement** capability. It allows the agent to learn from past experiences, avoid repeating mistakes, and improve performance on similar puzzles over time.

The system implements a **dual-layer architecture** combining structured storage with semantic search, enabling both precise data retrieval and intelligent pattern matching across thousands of past experiences.

---

## Architecture

The memory system consists of two coordinated layers:

### Layer 1: Episodic Memory
- **Purpose:** Structured storage of all agent experiences
- **Storage:** JSON-based persistence
- **Strengths:** Fast exact queries, field-based filtering, data integrity
- **Use Case:** Retrieve recent failures, get episode by ID, statistical analysis

### Layer 2: Vector Memory
- **Purpose:** Semantic search across experiences
- **Storage:** ChromaDB with OpenAI embeddings
- **Strengths:** Conceptual similarity matching, fuzzy retrieval
- **Use Case:** Find similar past failures, identify recurring patterns

### Coordination Layer
- **Purpose:** Unified API for both layers
- **Pattern:** Facade design pattern
- **Strengths:** Simplified interface, automatic synchronization, error handling
- **Use Case:** All agent interactions with memory

---

## Algorithm

### Experience Storage Flow

```
Agent solves puzzle
    ↓
System evaluates success/failure
    ↓
Agent performs self-reflection
    ↓
MemoryStore.add_episode() called
    ↓
Single Episode object created (with unique ID)
    ↓
Stored in EpisodicMemory (JSON)
    ↓
Stored in VectorMemory (ChromaDB with embedding)
    ↓
Both layers synchronized via episode_id
```

### Semantic Search Flow

```
New puzzle arrives
    ↓
MemoryStore.search_similar_failures() called
    ↓
VectorMemory converts query to embedding (1536 dimensions)
    ↓
Cosine similarity computed against all stored episodes
    ↓
Top-k most similar failures retrieved
    ↓
MemoryStore enriches results with full details from EpisodicMemory
    ↓
Agent receives context: "You made these mistakes before, avoid them"
    ↓
Agent solves puzzle with improved reasoning
```

### Key Algorithms

**1. Dual-Write Consistency**
- Single Episode object created in MemoryStore
- Same object passed to both storage layers
- Guarantees episode_id consistency across layers

**2. Semantic Similarity**
```
text = puzzle_text + reflection + outcome
embedding = OpenAI(text-embedding-3-small)(text)
similarity_score = 1 - (cosine_distance / 2)
```

**3. Graceful Degradation**
```
if VectorMemory fails (API error):
    EpisodicMemory still saves data
    log warning
    continue operation
    sync later when API available
```

**4. Memory Synchronization**
```
for episode in EpisodicMemory:
    if episode_id not in VectorMemory:
        add to VectorMemory
        increment counter
return {added: count, errors: count}
```

---

## Data Model

### Episode Structure

```python
{
    "episode_id": "uuid-unique-identifier",
    "puzzle_id": "puzzle_001",
    "puzzle_text": "Complete puzzle description",
    "reasoning_path": "Agent's step-by-step reasoning",
    "final_answer": "Final answer provided",
    "outcome": "success | failure",
    "reflection": "Self-analysis of why succeeded/failed",
    "timestamp": "ISO 8601 datetime"
}
```

### Metadata (Vector Storage)

```python
{
    "puzzle_id": "puzzle_001",
    "outcome": "failure",
    "timestamp": "2024-01-15T10:30:00",
    "final_answer": "First 100 chars..."
}
```

---

## Usage

### Basic Operations

```python
from src.memory.memory_store import MemoryStore

# Initialize
memory = MemoryStore(
    episodic_storage_path="data/memory/episodes.json",
    vector_db_path="data/memory/vector_db"
)

# Store experience
episode = memory.add_episode(
    puzzle_id="logic_001",
    puzzle_text="If all A's are B's and all B's are C's, are all A's C's?",
    reasoning_path="I assumed A is the largest...",
    final_answer="No",
    outcome="failure",
    reflection="I forgot the transitivity rule. If A⊂B and B⊂C, then A⊂C."
)

# Search for similar past failures
similar_failures = memory.search_similar_failures(
    query_text="If all X's are Y's and all Y's are Z's...",
    n_results=3
)

# Use in prompt
prompt = "Learn from these past mistakes:\n"
for failure in similar_failures:
    prompt += f"\nSimilarity: {failure['similarity_score']:.2f}\n"
    prompt += f"Mistake: {failure['episode'].reflection}\n"

# Get statistics
stats = memory.get_stats()
print(f"Success rate: {stats['episodic_memory']['success_rate']:.2f}%")
```

### Advanced Operations

```python
# Sync memories (if vector DB corrupted)
result = memory.sync_memories()
print(f"Synced: {result['added']} episodes, {result['errors']} errors")

# Get recent failures (without semantic search)
recent = memory.get_recent_failures(limit=5)

# Retrieve specific episode
episode = memory.get_episode_by_id("uuid-here")

# Clear all memory
memory.clear()
```

---

## Error Handling

### Scenario: OpenAI API Unavailable

```
1. VectorMemory.add_episode() raises exception
2. MemoryStore catches exception
3. Logs warning: "Failed to add to vector memory"
4. Episode still saved in EpisodicMemory
5. Agent continues operation
6. Later: sync_memories() restores vector layer
```

### Scenario: Corrupted Vector Database

```
1. VectorMemory.clear() called (or DB corrupted)
2. EpisodicMemory still intact
3. Call sync_memories()
4. All episodes re-embedded and re-added to VectorMemory
5. System fully restored
```

---

## Performance Characteristics

| Operation | EpisodicMemory | VectorMemory | Combined |
|-----------|----------------|--------------|----------|
| Add episode | O(1) | O(n) for embedding | O(n) |
| Search by ID | O(n) | O(1) | O(1) |
| Semantic search | N/A | O(log n) | O(log n) |
| Get recent failures | O(n) | N/A | O(n) |
| Clear all | O(1) | O(1) | O(1) |

**Scalability:**
- EpisodicMemory: Handles 10,000+ episodes efficiently
- VectorMemory: ChromaDB optimized for millions of vectors
- Combined: Suitable for large-scale experiments (thousands of puzzles)

---

## Design Decisions

### Why Dual-Layer Architecture?

**Problem:** Need both exact queries (get by ID, filter by outcome) and semantic search (find similar experiences)

**Solution:** 
- JSON for structured queries (fast, simple, debuggable)
- ChromaDB for semantic search (intelligent, fuzzy matching)
- Facade layer to coordinate both

**Trade-offs:**
- ✅ Flexibility: Can use either layer independently
- ✅ Reliability: If one fails, other continues
- ✅ Debuggability: JSON file human-readable
- ⚠️ Complexity: Two storage systems to maintain
- ⚠️ Cost: OpenAI embeddings API calls

### Why Pydantic Models?

**Problem:** LLM outputs are unpredictable, need validation

**Solution:** Pydantic BaseModel with strict type checking

**Benefits:**
- Runtime validation prevents corrupted data
- Easy JSON serialization
- Integration with LLM function calling
- Self-documenting code

### Why ChromaDB over FAISS?

**Problem:** Need persistent vector storage with metadata filtering

**Solution:** ChromaDB (not FAISS)

**Reasons:**
- Built-in persistence (no manual save/load)
- Metadata filtering (filter by outcome)
- Simpler API for LLM applications
- Active maintenance and community

---

## Limitations

1. **Embedding Cost:** OpenAI API calls cost money (~$0.0001 per episode)
2. **Latency:** Embedding generation adds ~200ms per episode
3. **Model Dependency:** Quality depends on embedding model
4. **No Incremental Learning:** Agent doesn't update model weights

## Future Enhancements

1. **Local Embeddings:** Replace OpenAI with Sentence-Transformers (free, faster)
2. **Advanced Database:** Migrate to PostgreSQL + pgvector
3. **Memory Compression:** Summarize old episodes to save space
4. **Active Learning:** Intelligently select which experiences to remember
5. **Multi-Modal Memory:** Store images, diagrams alongside text

---

## Summary

The memory system enables true self-improvement by:
- **Recording** every experience with structured data
- **Learning** from past mistakes through semantic search
- **Adapting** behavior based on retrieved experiences
- **Improving** performance over thousands of iterations

This is not just storage—it's the agent's **learning mechanism**.
