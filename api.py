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
        """
        Constructs a dynamic Cypher query based on LLM constraints
        to find the shortest semantic path in the Knowledge Graph.
        """
        rel_types = ["HAS_EXIT", "CONNECTS_TO"]
        
        # Check constraints safely without list comprehensions to avoid syntax UI bugs
        avoid_str = str(avoid).lower()
        if "stairs" not in avoid_str:
            rel_types.append("CONNECTS_VIA_STAIRS")
        if "elevator" not in avoid_str and "lift" not in avoid_str:
            rel_types.append("CONNECTS_VIA_ELEVATOR")
            
        rel_string = "|".join(rel_types)

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
        return None
    
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

    SYSTEM_PROMPT = """You are the spatial reasoning brain for a navigation assistant at DTU.
RULES:
1. Decide intent: "navigate" or "info".
2. If "navigate": set "target" to the destination name. Prefer EXACT words.
3. If "info": write a conversational answer in "reply" using the Context provided.
4. "avoid" is a list of hazard types to exclude (e.g., ['stairs', 'elevator']).
Respond ONLY with valid JSON - no extra text:
{
  "intent":  "navigate",
  "target":  "destination name",
  "avoid":   [],
  "reply":   ""
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

    # 3. Fuzzy Match Target
    target_str = llm_intent.get("target", "")
    hazards_to_avoid = llm_intent.get("avoid", [])

    synonyms = {"library": "bibliotek", "canteen": "kantine", "food": "kantine", "cafe": "kantine"}
    target_str = synonyms.get(target_str.lower(), target_str)

    best_match = target_str
    if state.valid_names:
        matches = difflib.get_close_matches(target_str, state.valid_names, n=1, cutoff=0.45)
        if matches: best_match = matches[0]

    # 4. Neo4j Cypher Execution
    route_nodes = await state.db.find_semantic_route(current_id, best_match, hazards_to_avoid)

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)