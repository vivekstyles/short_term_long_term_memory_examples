# Agentic Memory Systems: From Working Context to Long-Term Cognition

A progressive, hands-on Python curriculum and production-grade reference architecture illustrating how **Short-Term Memory** and **Long-Term Memory** function in modern autonomous agentic AI systems.

Powered locally by **Ollama** (`gemma4:12b`, `qwen3.8:27b`) with zero mandatory external API subscriptions, alongside a built-in deterministic simulation engine and high-speed local embedding vector search.

---

## 1. Conceptual Foundations: The Cognitive Architecture of Agentic Memory

In human cognition and computer architecture alike, memory is not a monolithic block—it is a **stratified hierarchy** balanced across speed, capacity, persistence, and retrieval cost.

```
       +---------------------------------------------------------+
       |           IN-CONTEXT SHORT-TERM WORKING MEMORY          |
       |  Fast Attention | Bounded Context Window | Volatile RAM |
       |                                                         |
       |  +------------------------+  +------------------------+ |
       |  | Level 1: Sliding Window|  | Level 2: Scratchpad &  | |
       |  | FIFO / Token Budgeting |  | Summary Compression    | |
       |  +------------------------+  +------------------------+ |
       +----------------------------+----------------------------+
                                    |
          Active Tool Retrieval     |   Offline Consolidation (Sleep)
          (Vector Search, Hybrid)   |   (Fact Extraction, Pruning)
                                    v
       +---------------------------------------------------------+
       |             PERSISTENT LONG-TERM EXTERNAL MEMORY        |
       |  Infinite Storage | Out-of-Context | Disk / SQLite / Vec|
       |                                                         |
       |  +------------------------+  +------------------------+ |
       |  | Level 3: Semantic      |  | Level 4: Episodic      | |
       |  | Facts & User Profiles  |  | Stanford Memory Stream | |
       |  +------------------------+  +------------------------+ |
       |  +------------------------+  +------------------------+ |
       |  | Level 5: Procedural    |  | Level 6: Unified MemGPT| |
       |  | Skills & Trajectories  |  | Core Memory Tools      | |
       |  +------------------------+  +------------------------+ |
       +---------------------------------------------------------+
```

### The Computer Architecture Analogy
- **CPU Registers & L1/L2 Cache** $\leftrightarrow$ **In-Context System Prompt & Active Tokens**: Immediate access during model forward pass, but strictly limited by maximum context length (e.g. 8k, 32k, 128k tokens).
- **System RAM** $\leftrightarrow$ **Session Buffer & Summaries**: Active conversation turns, rolling context summaries, and scratchpad blackboard state.
- **Disk / NVMe SSD / Database** $\leftrightarrow$ **External Long-Term Stores**: SQLite relational databases, dense vector embedding indexes, and procedural skill registries that survive across session restarts.

---

## 2. Comparison Matrix: Short-Term vs. Long-Term Memory

| Attribute | Short-Term Memory (STM) | Long-Term Memory (LTM) |
| :--- | :--- | :--- |
| **Storage Location** | Directly inside the LLM prompt context window | External database (SQLite, Vector Store, JSON) |
| **Capacity** | Strictly bounded (tokens) | Virtually unlimited (gigabytes / terabytes) |
| **Persistence** | Ephemeral (lost when session ends or window slides) | Permanent (persists across restarts and years) |
| **Retrieval Mechanism**| Direct Transformer self-attention across tokens | Semantic vector search, BM25 keyword, SQL queries |
| **Cognitive Role** | Working state, immediate instructions, dialogue flow | Knowledge, historical experiences, learned skills |
| **Update Mechanism** | Token concatenation, sliding window, rolling summary | Vector insertion, importance scoring, consolidation |

---

## 3. Curriculum & Progression Roadmap

### Level 1: Message Buffer & Sliding Window (Working Context)
*File: `src/short_term/buffer_window.py` | Example: `examples/01_buffer_and_sliding_window.py`*
- **Naive Buffer**: Preserves all turns sequentially. Quickly crashes against context token limits and increases inference costs.
- **Sliding Window (FIFO)**: Enforces hard limits via turn count ($K$ turns) or token budgets ($T$ tokens). Oldest non-anchor turns are evicted as new turns arrive.
- **Anchor Preservation**: Ensures critical foundational instructions (e.g., `role == Role.SYSTEM` or messages tagged `pinned=True`) are **never evicted**, maintaining agent identity and core guardrails.

### Level 2: Conversation Summary Buffer & Working Scratchpad
*File: `src/short_term/summary_scratchpad.py` | Example: `examples/02_summary_and_scratchpad.py`*
- **Rolling Summarization**: When token limits are approached, older turns are not discarded—the LLM condenses them into an evolving, information-dense summary while retaining recent turns verbatim.
- **Working Scratchpad (Cognitive Blackboard)**: Decouples the conversation transcript from the agent's internal operational state. Maintains active goals, subtask status checklists (`[PENDING]`, `[IN_PROGRESS]`, `[COMPLETED]`), runtime variables, and internal tool notes.

### Level 3: Semantic / Declarative Memory (Knowledge & Facts)
*File: `src/long_term/semantic_memory.py` | Example: `examples/03_semantic_memory.py`*
- **Persistent Fact Store**: Durable, timeless facts stored in SQLite (user preferences, system configurations, domain constraints).
- **Dense Vector Embeddings**: Enables conceptual retrieval (e.g., matching `"dark theme"` when the user asks about `"editor appearance"`).
- **Hybrid Search**: Combines structured relational filtering (`WHERE entity = 'user' AND category = 'preference'`) with cosine vector ranking.

### Level 4: Episodic Memory Stream & Reflection
*File: `src/long_term/episodic_memory.py` | Example: `examples/04_episodic_memory.py`*
- **Stanford Generative Agents Model (Park et al., 2023)**: A chronological stream of specific events, actions, and observations.
- **Tri-Partite Retrieval Scoring Formula**:
  $$\text{Score}(m, q) = w_{rec} \cdot \text{Recency}(m) + w_{imp} \cdot \text{Importance}(m) + w_{rel} \cdot \text{Relevance}(m, q)$$
  - $\text{Recency}(m) = \text{decay\_factor}^{(\Delta t / \text{hour})}$ (exponential temporal forgetting curve).
  - $\text{Importance}(m) \in [0, 1]$ (rated by LLM on a 1-10 scale).
  - $\text{Relevance}(m, q) \in [0, 1]$ (cosine similarity between query and memory embedding).
- **Cognitive Reflection**: The agent analyzes recent episodes, extracts generalized insights, and writes reflections back into the stream with high importance scores.

### Level 5: Procedural Memory & Dynamic Skills
*File: `src/long_term/procedural_memory.py` | Example: `examples/05_procedural_memory.py`*
- **Skill Library (Voyager Pattern)**: Reusable tool invocation routines, code recipes, and action procedures indexed with semantic trigger patterns and success metrics.
- **Few-Shot Trajectory Store**: Past successful execution traces retrieved dynamically as in-context exemplars when a similar new goal arrives.
- **Dynamic Guidance**: Injects only the relevant skills and trajectory exemplars into the agent's prompt, avoiding prompt bloat.

### Level 6: Unified Cognitive Architecture (MemGPT Pattern & Consolidation)
*File: `src/unified/memgpt_agent.py` & `src/unified/consolidation.py` | Example: `examples/06_unified_cognitive_agent.py`*
- **MemGPT Hierarchical Memory**:
  - **Core Memory** (In-context block: `Human Profile` + `Agent Persona`).
  - **Archival Memory** (Out-of-context semantic vector database).
  - **Working Context** (Bounded sliding window).
- **Autonomous Memory Tools**:
  - `core_memory_append(section, content)`
  - `core_memory_replace(section, old_str, new_str)`
  - `archival_memory_insert(content, category)`
  - `archival_memory_search(query)`
- **End-of-Session Memory Consolidation ('Sleep Cycle')**:
  - At the end of a session, a consolidation engine digests volatile short-term chat transcripts.
  - Extracts durable facts into `SemanticMemory` and session events into `EpisodicMemory`.
  - Clears the working message buffer so the next session starts fresh with clean tokens, while retaining all knowledge in long-term storage!

---

## 4. Local Ollama Integration

The project is configured to run with your local **Ollama** instance (`http://localhost:11434`):
- **Default Model**: `gemma4:12b` (or `qwen3.8:27b` via `.env`).
- **Resilient Fallback**: If an Ollama CPU call times out or Ollama is offline, the agent falls back to the deterministic local engine, ensuring uninterrupted execution.
- **Zero-Dependency Vector Embeddings**: Uses a built-in subword n-gram TF-IDF cosine embedding engine in `src/core/embeddings.py` that computes similarity in $\approx 0.1\text{ ms}$ without requiring external embedding API endpoints.

---

## 5. Project Structure

```
short_term_long_term_memory_examples/
├── README.md                           # Complete theoretical and practical guide
├── pyproject.toml                      # Project metadata and pytest configuration
├── requirements.txt                    # Project dependencies
├── .env.example                        # Template for environment variables
├── .env                                # Local configuration (Ollama model & URL)
├── main.py                             # Master CLI entrypoint
├── src/
│   ├── __init__.py
│   ├── visualizer.py                  # Rich console formatting & memory inspection
│   ├── cli.py                         # Interactive menu and live chat inspector
│   ├── core/
│   │   ├── message.py                 # Message models (User, Assistant, System, Tool) & token counter
│   │   ├── embeddings.py              # Zero-dependency dense vector embedding engine
│   │   └── llm_backend.py             # OllamaLLM + MockLLM with resilient fallback
│   ├── short_term/
│   │   ├── buffer_window.py           # Level 1: Buffer, FIFO & token-bounded sliding window
│   │   └── summary_scratchpad.py      # Level 2: Conversation summary buffer & working scratchpad
│   ├── long_term/
│   │   ├── semantic_memory.py         # Level 3: Fact storage, user profiling & SQLite vector search
│   │   ├── episodic_memory.py         # Level 4: Experience stream, Stanford scoring & reflection
│   │   └── procedural_memory.py       # Level 5: Dynamic skills, workflows & few-shot trajectory store
│   └── unified/
│       ├── memgpt_agent.py            # Level 6: MemGPT-style agent with core/archival memory tools
│       └── consolidation.py           # Sleep cycle (short-to-long term memory migration)
├── examples/
│   ├── 01_buffer_and_sliding_window.py# Standalone script: Level 1
│   ├── 02_summary_and_scratchpad.py   # Standalone script: Level 2
│   ├── 03_semantic_memory.py          # Standalone script: Level 3
│   ├── 04_episodic_memory.py          # Standalone script: Level 4
│   ├── 05_procedural_memory.py        # Standalone script: Level 5
│   └── 06_unified_cognitive_agent.py  # Standalone script: Level 6
└── tests/
    ├── __init__.py
    ├── test_short_term.py             # Pytest suite for Short-Term memory
    ├── test_long_term.py              # Pytest suite for Long-Term memory
    └── test_unified.py                # Pytest suite for Unified architecture
```

---

## 6. Quickstart & Installation

### 1. Initialize Virtual Environment & Dependencies
```bash
# Clone or navigate to the directory
cd short_term_long_term_memory_examples

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Local Ollama
Ensure Ollama is running:
```bash
ollama list
```
*(Your `.env` is already configured for `OLLAMA_MODEL=gemma4:12b` on `http://localhost:11434`)*

---

## 7. Running Demonstrations & Live Inspector

### Interactive Master Menu
Launch the interactive terminal curriculum:
```bash
python main.py
```

### Run Standalone Demonstrations
You can run any level directly:
```bash
python examples/01_buffer_and_sliding_window.py
python examples/02_summary_and_scratchpad.py
python examples/03_semantic_memory.py
python examples/04_episodic_memory.py
python examples/05_procedural_memory.py
python examples/06_unified_cognitive_agent.py
```

Or via CLI flags:
```bash
# Run Level 1
python main.py --demo 1

# Run all 6 levels sequentially with instant Mock LLM
python main.py --demo all --mock
```

### Live Interactive Memory Chat Inspector
Chat directly with the cognitive agent and inspect its memory blocks mutating live in the terminal:
```bash
python main.py --interactive
```
During the chat:
- Type `/memory` to inspect the agent's active Core Memory and working buffer.
- Type `/consolidate` to trigger the offline sleep cycle (migrating short-term dialogue to long-term storage).
- Type `/quit` to exit.

---

## 8. Running Automated Tests

Run the complete test suite with `pytest`:
```bash
pytest tests/ -v
```
All 11 tests execute in $<1$ second covering:
- Token budgeting and sliding window anchor preservation.
- Conversation summarization triggers and verbatim turn retention.
- Working memory scratchpad updates.
- Semantic fact CRUD, SQLite persistence, and dense vector search.
- Stanford episodic scoring equations and reflection loops.
- Procedural skill registration and trajectory retrieval.
- MemGPT Core Memory self-editing tools.
- Memory consolidation sleep cycles.
