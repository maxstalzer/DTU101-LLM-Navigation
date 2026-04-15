import os
import json
import math
import requests
import difflib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

# Load Environment Variables
load_dotenv()
CAMPUS_AI_KEY = os.getenv("CAMPUS_AI_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password") # Update to match your DB

if not CAMPUS_AI_KEY:
    print("CRITICAL ERROR: CAMPUS_AI_KEY not found in .env file.")
    exit()

def mercator_to_latlon(x, y):
    lon = (x / 20037508.34) * 180
    lat = (y / 20037508.34) * 180
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180)) - math.pi / 2)
    return lat, lon

def get_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

# ==========================================
# 1. Define Request/Response Models
# ==========================================
class AgentQueryRequest(BaseModel):
    user_input: str
    current_node_id: str

class RouteCoordinate(BaseModel):
    lat: float
    lon: float
    name: str
    floor: float
    is_portal: bool

class AgentQueryResponse(BaseModel):
    intent: str
    reply: str
    target_name: str | None = None
    target_id: str | None = None
    instructions: list = []
    path_coordinates: list[RouteCoordinate] = []

# ==========================================
# 2. Neo4j Database Connector
# ==========================================
class Neo4jConnector:
    def __init__(self, uri, user, password):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        print("🔌 Connected to Neo4j Graph Database.")

    async def close(self):
        await self.driver.close()

    async def get_valid_room_names(self):
        """Fetches all known space names on boot for fuzzy matching."""
        query = "MATCH (s:Space) WHERE s.name IS NOT NULL RETURN DISTINCT s.name AS name"
        async with self.driver.session() as session:
            result = await session.run(query)
            names = [record["name"] async for record in result]
        return names

    async def get_room_context(self, node_id: str):
        """Queries Neo4j for the current room name and adjacent rooms."""
        query = """
        MATCH (s:Space {poi_id: $node_id})
        OPTIONAL MATCH (s)-[:HAS_EXIT]->(:Portal)-[:CONNECTS_TO]->(adj:Space)
        RETURN s.name AS current_name, s.floor AS floor, collect(DISTINCT adj.name) AS adjacent_rooms
        """
        async with self.driver.session() as session:
            result = await session.run(query, node_id=node_id)
            record = await result.single()
            if record:
                return record["current_name"], record["floor"], record["adjacent_rooms"]
        return "Unknown", 0.0, []

    async def find_semantic_route(self, start_id: str, target_name: str, avoid: list):
        rel_types = ["HAS_EXIT", "CONNECTS_TO"]
        avoid_str = str(avoid).lower()
        if "stairs" not in avoid_str:
            rel_types.append("CONNECTS_VIA_STAIRS")
        if "elevator" not in avoid_str and "lift" not in avoid_str:
            rel_types.append("CONNECTS_VIA_ELEVATOR")
        rel_string = "|".join(rel_types)

        # Primary: exact CONTAINS match
        query = f"""
        MATCH (start:Space {{poi_id: $start_id}})
        MATCH (end:Space) WHERE toLower(end.name) CONTAINS toLower($target_name)
        MATCH path = shortestPath((start)-[:{rel_string}*]-(end))
        RETURN [n IN nodes(path) | {{
            id: coalesce(n.poi_id, n.portal_id),
            name: coalesce(n.name, 'Portal'),
            type: labels(n)[0],
            x: n.x, y: n.y, floor: n.floor
        }}] AS route, length(path) AS total_hops
        ORDER BY total_hops ASC LIMIT 1
        """
        async with self.driver.session() as session:
            result = await session.run(query, start_id=start_id, target_name=target_name)
            record = await result.single()
            if record:
                return record["route"]

        # Fallback 1: individual meaningful words
        STOP_WORDS = {
            "the", "a", "an", "to", "get", "find", "go", "take", "me",
            "i", "want", "need", "please", "can", "you", "how", "where",
            "is", "are", "room", "floor", "building", "nearest", "closest"
        }
        words = [w for w in target_name.lower().split() if len(w) > 3 and w not in STOP_WORDS]
        for word in words:
            async with self.driver.session() as session:
                result = await session.run(query, start_id=start_id, target_name=word)
                record = await result.single()
                if record:
                    return record["route"]

        # Fallback 2: progressive prefix truncation (e.g. "glassale" → "glassal" → "glassa")
        # Useful when the stored name is a substring of what the user typed, or vice versa
        cleaned = target_name.strip().lower()
        if len(cleaned) > 5:
            for prefix_len in range(len(cleaned) - 1, 4, -1):
                prefix = cleaned[:prefix_len]
                async with self.driver.session() as session:
                    result = await session.run(query, start_id=start_id, target_name=prefix)
                    record = await result.single()
                    if record:
                        return record["route"]

        return None

    # Add this diagnostic helper to quickly check what the DB actually has:
    async def search_room_names(self, query_str: str):
        """Debug: find what names in the DB are close to a query string."""
        query = """
        MATCH (s:Space) WHERE toLower(s.name) CONTAINS toLower($q)
        RETURN s.name AS name, s.poi_id AS id LIMIT 10
        """
        async with self.driver.session() as session:
            result = await session.run(query, q=query_str)
            return [{"name": r["name"], "id": r["id"]} async for r in result]

    async def get_random_room_id(self):
            """Fetches a random valid room ID for initial frontend 'cold starts'."""
            query = "MATCH (s:Space) WHERE s.name IS NOT NULL RETURN s.poi_id AS poi_id LIMIT 50"
            async with self.driver.session() as session:
                result = await session.run(query)
                records = [record["poi_id"] async for record in result]
                import random
                return random.choice(records) if records else ""

# ==========================================
# 3. LLM & Turn-by-Turn Logic
# ==========================================
def query_navigation_intent(user_input, current_name, nearby_rooms):
    url = "https://chat.campusai.compute.dtu.dk/api/chat/completions"
    headers = {"Authorization": f"Bearer {CAMPUS_AI_KEY}", "Content-Type": "application/json"}
    nearby_str = ", ".join(nearby_rooms) if nearby_rooms else "None"

    SYSTEM_PROMPT = """You are the intent-extraction brain for a DTU indoor navigation system.
Your ONLY job is to classify the user's request and extract parameters.

CRITICAL RULES:
1. If the user wants to GO somewhere, reach somewhere, find a location, or asks HOW TO GET somewhere → intent MUST be "navigate", NEVER "info".
2. If the user asks a factual question NOT about going somewhere (e.g., "what floor am I on?", "what is this room?") → intent is "info".
3. For "navigate": set "target" to the EXACT destination name as the user said it. Do NOT paraphrase.
4. For "info": write a short factual reply using only the Context provided. NEVER give navigation directions in an info reply.
5. "avoid" is a list of mobility hazards: e.g., ["stairs"], ["elevator"], or [].

Examples:
- "how do I get to the canteen" → navigate
- "take me to glassalen" → navigate  
- "where is the bathroom" → navigate
- "what floor am I on" → info
- "what is this room used for" → info

Respond ONLY with valid JSON, no extra text:
{
  "intent": "navigate",
  "target": "destination name exactly as user said",
  "avoid": [],
  "reply": ""
}"""

    payload = {
        "model": "Gemma 4 (Chat)",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context: I am in {current_name}. Adjacent rooms are: {nearby_str}. User Request: {user_input}"}
        ],
        "stream": False
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10.0)
        response.raise_for_status()
        
        # Safe extraction without using string slicing brackets
        response_text = response.json().get('choices', [{'message': {'content': ''}}])[0].get('message', {}).get('content', '')
        response_text = response_text.replace("```json", "").replace("```", "").strip()
            
        return json.loads(response_text)
    
    except requests.exceptions.Timeout:
        return {"error": "timeout"}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": "parsing_error"}

def generate_turn_by_turn(route_nodes):
    """Generates instructions directly from the Neo4j Route array."""
    if not route_nodes or len(route_nodes) < 2:
        return ["You are already at your destination."]

    instructions = [f"🟢 Start at {route_nodes[0]['name']}."]
    current_heading = None

    for i in range(len(route_nodes) - 1):
        curr_node = route_nodes[i]
        next_node = route_nodes[i+1]
        
        x1, y1 = curr_node['x'], curr_node['y']
        x2, y2 = next_node['x'], next_node['y']
        
        # Vertical Transition
        if abs(curr_node['floor'] - next_node['floor']) >= 0.5:
            instructions.append(f"⚠️ Take the {next_node['type'].lower()} to floor {next_node['floor']}.")
            current_heading = None
            continue

        target_heading = get_angle(x1, y1, x2, y2)
        distance = math.hypot(x2 - x1, y2 - y1)
        paces = max(1, int(distance / 0.75))

        if distance < 1.0 and next_node['type'] != 'Portal':
            continue

        if next_node['type'] == 'Portal':
            verb = "Pass through the doorway"
            dest = "the next area"
            if i + 2 < len(route_nodes):
                dest = route_nodes[i+2]['name']
            action_str = f"🚶 {verb} towards {dest} (~{paces} paces)."
        else:
            action_str = f"🚶 Walk across to {next_node['name']} (~{paces} paces)."

        if current_heading is not None:
            turn_angle = (target_heading - current_heading + 180) % 360 - 180
            if abs(turn_angle) < 20: turn_cmd = "Continue straight"
            elif 20 <= turn_angle < 75: turn_cmd = "Take a slight left"
            elif 75 <= turn_angle < 120: turn_cmd = "Turn left"
            elif 120 <= turn_angle <= 180: turn_cmd = "Take a sharp left"
            elif -75 < turn_angle <= -20: turn_cmd = "Take a slight right"
            elif -120 < turn_angle <= -75: turn_cmd = "Turn right"
            else: turn_cmd = "Take a sharp right"
            
            action_str = f"🔄 {turn_cmd}, then " + action_str[2:].lower()

        instructions.append(action_str)
        current_heading = target_heading

    instructions.append(f"🏁 You have arrived at {route_nodes[-1]['name']}.")
    return instructions

# ==========================================
# 4. Global State & FastAPI Boot
# ==========================================
class AppState:
    db: Neo4jConnector = None
    valid_names: list = []

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Booting up Semantic Campus AI Backend...")
    state.db = Neo4jConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        state.valid_names = await state.db.get_valid_room_names()
        print(f"✅ Loaded {len(state.valid_names)} spatial semantic concepts from Neo4j.")
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        print("Please ensure Neo4j is running and credentials in .env are correct.")
        
    yield
    print("🛑 Shutting down and disconnecting from Graph...")
    if state.db:
        await state.db.close()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 5. The Core Endpoint
# ==========================================
@app.post("/api/v1/agent/query", response_model=AgentQueryResponse)
async def process_agent_query(request: AgentQueryRequest):
    current_id = request.current_node_id
    user_input = request.user_input

    if not current_id:
        current_id = await state.db.get_random_room_id()

    # 1. Fetch live context directly from the Knowledge Graph
    current_name, current_floor, nearby_rooms = await state.db.get_room_context(current_id)
    if current_name == "Unknown":
        raise HTTPException(status_code=400, detail="Invalid starting node ID.")

    # 2. LLM Intent & Parameter Extraction
    llm_intent = query_navigation_intent(user_input, current_name, nearby_rooms)
    
    if not llm_intent or "error" in llm_intent:
        err_type = llm_intent.get("error", "unknown") if llm_intent else "unknown"
        if err_type == "timeout":
            raise HTTPException(status_code=504, detail="LLM Gateway Timeout.")
        else:
            raise HTTPException(status_code=502, detail=f"LLM API Error: {err_type}")

    if llm_intent.get("intent") == "info":
        return AgentQueryResponse(
            intent="info", 
            reply=llm_intent.get("reply", "Here is the requested info."),
            target_id=current_id 
        )

    # 3. Fuzzy Match / Name Resolution (tiered approach)
    target_str = llm_intent.get("target", "").strip()
    hazards_to_avoid = llm_intent.get("avoid", [])

    # Strip leading articles before synonym lookup
    ARTICLES = {"the ", "a ", "an ", "to ", "to the ", "a the "}
    target_lower = target_str.lower()
    for article in ARTICLES:
        if target_lower.startswith(article):
            target_lower = target_lower[len(article):]
            break

    # Expand known synonyms (Danish/English)
    SYNONYMS = {
        "library": "bibliotek",
        "canteen": "kantine",
        "food": "kantine",
        "cafe": "kantine",
        "cafeteria": "kantine",
        "auditorium": "glassalen",   # add known aliases
        "glass hall": "glassalen",
        "bathroom": "toilet",
        "restroom": "toilet",
        "toilet": "toilet",
        "elevator": "elevator",
        "lift": "elevator",
    }
    best_match = SYNONYMS.get(target_lower, target_str)

    # Strategy 1: Try the resolved name directly via Cypher CONTAINS (trust the DB)
    route_nodes = await state.db.find_semantic_route(current_id, best_match, hazards_to_avoid)

    # Strategy 2: If that fails, try fuzzy match against cached names
    if not route_nodes and state.valid_names:
        matches = difflib.get_close_matches(best_match, state.valid_names, n=3, cutoff=0.35)
        for candidate in matches:
            route_nodes = await state.db.find_semantic_route(current_id, candidate, hazards_to_avoid)
            if route_nodes:
                best_match = candidate
                break

    # Strategy 3: If still nothing, try the original raw user string (LLM may have paraphrased)
    if not route_nodes and best_match.lower() != target_str.lower():
        route_nodes = await state.db.find_semantic_route(current_id, target_str, hazards_to_avoid)
        if route_nodes:
            best_match = target_str

    if not route_nodes:
        return AgentQueryResponse(
            intent="error", 
            reply=f"Neo4j Routing Error: Cannot find a valid path to '{best_match}' while strictly avoiding {hazards_to_avoid}.",
            target_id=current_id 
        )

    target_id_final = route_nodes[-1]['id']
    target_name_final = route_nodes[-1]['name']

    # 5. Format Output
    turn_instructions = generate_turn_by_turn(route_nodes)
    
    path_coords = []
    for node in route_nodes:
        lat, lon = mercator_to_latlon(node['x'], node['y'])
        path_coords.append(RouteCoordinate(
            lat=lat, 
            lon=lon, 
            name=node['name'], 
            floor=float(node.get('floor', 0)),
            is_portal=(node['type'] == 'Portal')
        ))

    return AgentQueryResponse(
        intent="navigate",
        reply=llm_intent.get("reply", f"Here are the semantic directions to {target_name_final}."),
        target_name=target_name_final,
        target_id=target_id_final,
        instructions=turn_instructions,
        path_coordinates=path_coords
    )

@app.get("/api/v1/debug/search")
async def debug_search(q: str):
    """Quick diagnostic: what does Neo4j actually have for a name fragment?"""
    results = await state.db.search_room_names(q)
    fuzzy = difflib.get_close_matches(q, state.valid_names, n=5, cutoff=0.3)
    return {
        "cypher_contains_matches": results,
        "fuzzy_matches": fuzzy,
        "total_cached_names": len(state.valid_names)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)