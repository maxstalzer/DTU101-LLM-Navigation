# LLM-Based Semantic Indoor Navigation & Orientation System for the Visually Impaired

**Background** Navigating complex indoor campus environments is a significant challenge for Visually Impaired Persons. While DTU utilizes MazeMap for indoor routing, standard A-to-B coordinate routing may be insufficient for those who rely on Orientation & Mobility (O&M) cues—such as relative headings, step counts and hazard awareness (e.g. stairs or steep ramps). Furthermore, standard routing systems lack semantic understanding of spaces (e.g., knowing that a "canteen" is a noisy area, or that two rooms share a physical doorway).

**The Idea** This project aims to build an AI-driven, semantic navigation agent for DTU Building 101. By extracting raw geometric polygon data from MazeMap, we can construct a Semantic Knowledge Graph mapping the physical campus to the **BOT (Building Topology Ontology)**.

An LLM will act as the spatial reasoning engine. If a user asks, _"How do I get to a quiet room from the Library, avoiding stairs?"_, the LLM will interpret the intent, anchor the room name to the Knowledge Graph and apply hazard constraints. The underlying system calculates a physical path through the polygons and translates the vectors into safe, human-scale O&M instructions (e.g., _"Exit the Library, turn right and walk 15 paces"_).

**Methodology & Challenges** Creating a true single source of truth for spatial navigation is not trivial. A primary challenge is "state-drift" between static routing meshes and semantic databases. To solve this, the project dynamically builds an in-memory physical topological graph (NetworkX) directly from the semantic triples (`rdflib`) at runtime using spatial mathematics (`Shapely`).

Furthermore, LLMs notoriously hallucinate spatial geometry. Instead of relying on the LLM to do continuous-space routing, this architecture utilizes a Hybrid GraphRAG approach: the LLM extracts constraints and translates output text, while a deterministic graph engine handles the physical pathfinding and doorway portal generation.

**Proposed Endpoints (API Design)**

The system can be exposed via a backend API (e.g., base path `/api/v1`) to serve web frontends or voice-assisted mobile apps.

- **`GET /rooms/{room_id}/context`**
    
    - Fetches the semantic context of a space (adjacent rooms, hazards, spatial footprint).
        
- **`GET /rooms/search`**
    
    - Fuzzy-matches natural language queries (e.g., "food", "main entrance") to precise graph Node IDs.
        
- **`POST /navigation/route`**
    
    - Calculates the shortest safe path between two points given an array of hazard constraints (e.g., ``).
        
- **`POST /navigation/om-guidance`**
    
    - Takes a physical coordinate path and translates it into step-by-step Orientation & Mobility instructions (paces, clock-face turns, floor transitions).
        
- **`POST /agent/query`**
    
    - The primary LLMOps endpoint. Accepts a natural language user query and current location, runs the GraphRAG pipeline, and returns the interpreted intent, the exact route, and a conversational audio-ready response.