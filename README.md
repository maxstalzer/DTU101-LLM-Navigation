# DTU Semantic Navigation Assistant

An AI-powered, accessibility-aware indoor navigation system for DTU Building 101. This project shifts spatial routing away from imperative Python scripts and into a **Semantic Knowledge Graph (Neo4j)**, using an LLM to translate natural language constraints into native Cypher queries (Text-to-Graph).

## Architecture Overview
1. **Data Pipeline:** Extracts raw spatial geometries from MazeMap and triplifies them into a formal Knowledge Graph.
2. **Space Syntax Engine:** Mathematically calculates wall intersections using `Shapely` to inject `Portal` (doorway) nodes, preventing "zig-zag" routing bugs in large open rooms.
3. **Semantic Router (FastAPI):** An LLMOps backend that uses Gemma 3 to dynamically generate Neo4j routing queries based on user constraints (e.g., avoiding stairs for accessibility).
4. **Interactive UI (Streamlit):** A chat interface with an embedded, interactive Plotly map rendering the Neo4j route over building blueprints.

---

## Prerequisites & Setup

1. **Neo4j Desktop:** Install and create a local DBMS. Ensure it is running on port `7687`.
2. **Environment Variables:** Create a `.env` file in the root directory:
```env
CAMPUS_AI_KEY=your_dtu_campus_ai_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
```
3. **Dependencies:** Install the required Python packages using `uv` or `pip`:
```bash
uv pip install requests shapely networkx neo4j fastapi uvicorn pydantic python-dotenv streamlit plotly
```

---

## The Pipeline: From Raw Data to Live App

Follow these steps in order to rebuild the graph and launch the application.

### Step 1: Mine the Spatial Data
Extracts raw POIs from the MazeMap API, filters out deleted nodes, and categorizes rooms with semantic labels (e.g., Canteens, Accessible Restrooms, Elevators).
* **Script:** `extract_mazemap.py`
* **Output:** `dtu_semantic_nodes.json`
* **Run:** `python extract_mazemap.py`

### Step 2: Generate the Knowledge Graph
Acts as a Semantic Triplification Engine. It reads the JSON, uses `Shapely` to find overlapping walls (within a 4.0m tolerance), injects `Portal` nodes at the boundaries, and adds 25-meter penalties to room-crossings to enforce realistic human walking paths.
* **Script:** `generate_cypher_graph.py`
* **Output:** `dtu_neo4j_seed.cypher` (A massive ~8,000 line database seed script).
* **Run:** `python generate_cypher_graph.py`

### Step 3: Import into Neo4j
Bypasses the Neo4j Browser UI limits by streaming the massive `.cypher` file directly into your local Neo4j database chunk-by-chunk.
* **Script:** `import_graph.py`
* **Run:** `python import_graph.py`
> *Verify:* Open Neo4j Browser and run `MATCH (n) RETURN n LIMIT 300` to see your graph.

### Step 4: Start the Semantic Router (Backend)
Boots the FastAPI server. It connects to Neo4j on startup to cache spatial concepts. When queried, it prompts the LLM to identify hazards, builds a Cypher `shortestPath` query that physically excludes those hazards, and returns the path coordinates.
* **Script:** `api.py`
* **Run:** `uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload`

### Step 5: Launch the UI (Frontend)
Starts the Streamlit chat application. It manages session state, renders the chat interface, and draws the interactive Plotly map with hoverable room centers and the rendered route.
* **Script:** `app.py`
* **Run:** `uv run streamlit run app.py`
