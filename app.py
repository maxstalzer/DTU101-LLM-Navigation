import streamlit as st
import requests
import json
import math
import plotly.graph_objects as go

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000/api/v1/agent/query"
st.set_page_config(page_title="DTU Navigation", layout="wide")

# --- GEOMETRY HELPERS ---
def mercator_to_latlon(x, y):
    """Converts MazeMap Web Mercator to GPS Lat/Lon for Plotly."""
    lon = (x / 20037508.34) * 180
    lat = (y / 20037508.34) * 180
    lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180)) - math.pi / 2)
    return lat, lon

def convert_coords_to_latlon(coords):
    """Recursively converts nested GeoJSON coordinate arrays."""
    if not coords:
        return []
    # If we hit the innermost pair [x, y], convert and return [lon, lat]
    if isinstance(coords[0], (int, float)): 
        lat, lon = mercator_to_latlon(coords[0], coords[1])
        return [lon, lat] 
    # Otherwise, keep digging into the nested lists
    return [convert_coords_to_latlon(c) for c in coords]

@st.cache_data
def load_room_polygons(filepath="data/dtu_semantic_nodes.json"):
    """Loads and caches the room polygons as a valid GeoJSON FeatureCollection."""
    geojson_features = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            nodes_data = json.load(f)

        for node_id, attr in nodes_data.items():
            if '101' not in str(attr.get('building_name', '')):
                continue
            
            poly_geom = attr.get('polygon')
            if poly_geom and poly_geom.get('type') in ['Polygon', 'MultiPolygon']:
                # Convert all Web Mercator coordinates to Lat/Lon
                latlon_coords = convert_coords_to_latlon(poly_geom['coordinates'])
                
                geojson_features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": poly_geom['type'],
                        "coordinates": latlon_coords
                    },
                    "properties": {"name": attr.get('name', 'Unknown')}
                })
        return {"type": "FeatureCollection", "features": geojson_features}
    except FileNotFoundError:
        st.warning(f"Could not find {filepath} to render room outlines.")
        return {"type": "FeatureCollection", "features": []}

# Load the polygons once into memory
ROOM_GEOJSON = load_room_polygons()

@st.cache_data
def load_room_data(filepath="data/dtu_semantic_nodes.json"):
    """Loads room polygons and center points for hover interactions."""
    geojson_features = []
    centers = {"lat": [], "lon": [], "name": []}
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            nodes_data = json.load(f)

        for node_id, attr in nodes_data.items():
            if '101' not in str(attr.get('building_name', '')):
                continue
            
            name = attr.get('name', 'Unknown')
            
            # 1. Extract Polygon GeoJSON
            poly_geom = attr.get('polygon')
            if poly_geom and poly_geom.get('type') in ['Polygon', 'MultiPolygon']:
                latlon_coords = convert_coords_to_latlon(poly_geom['coordinates'])
                geojson_features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": poly_geom['type'],
                        "coordinates": latlon_coords
                    },
                    "properties": {"name": name}
                })
            
            # 2. Extract Center Points for Hover
            cx, cy = attr.get('center_x'), attr.get('center_y')
            if cx is not None and cy is not None:
                c_lat, c_lon = mercator_to_latlon(cx, cy)
                centers["lat"].append(c_lat)
                centers["lon"].append(c_lon)
                centers["name"].append(f"<b>{name}</b>")

        return {"type": "FeatureCollection", "features": geojson_features}, centers
    except FileNotFoundError:
        st.warning(f"Could not find {filepath} to render room outlines.")
        return {"type": "FeatureCollection", "features": []}, centers

# Load the data once into memory
ROOM_GEOJSON, ROOM_CENTERS = load_room_data()

# --- SESSION STATE ---
if "current_node_id" not in st.session_state:
    st.session_state.current_node_id = "" 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_map" not in st.session_state:
    st.session_state.current_map = None
if "current_instructions" not in st.session_state:
    st.session_state.current_instructions = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ System Status")
    st.success("API Connected: http://localhost:8000")
    st.text_input("Current Location ID (Debug)", value=st.session_state.current_node_id, disabled=True)
    if st.button("Reset Session"):
        st.session_state.chat_history = []
        st.session_state.current_map = None
        st.session_state.current_instructions = []
        st.session_state.current_node_id = ""
        st.rerun()

# --- MAIN UI ---
st.title("DTU Orientation & Mobility Agent")
st.markdown("Ask for directions. The agent will calculate a safe, turn-by-turn route.")

col1, col2 = st.columns(2)

with col1:
    if st.session_state.current_map:
        st.plotly_chart(st.session_state.current_map, width='stretch')
    else:
        st.info("Map will appear here once a route is generated.")

with col2:
    st.subheader("🚶 O&M Guidance")
    if st.session_state.current_instructions:
        for step in st.session_state.current_instructions:
            st.write(step)
    else:
        st.write("Awaiting destination...")

st.divider()

# --- CHAT INTERFACE ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Where do you want to go? (e.g., 'Take me to the canteen')")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Calculating safe route via Neo4j..."):
        try:
            payload = {
                "user_input": user_input,
                "current_node_id": st.session_state.current_node_id
            }
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            data = response.json()

            agent_reply = data.get("reply", "Done.")

            if data.get("intent") == "navigate" and data.get("path_coordinates"):
                coords = data["path_coordinates"]
                lats  = [pt["lat"] for pt in coords]
                lons  = [pt["lon"] for pt in coords]
                texts = [pt["name"] for pt in coords]

                # Build the Plotly Map
                
                # Trace 1: Invisible hover zones (small dots) at the center of every room
                rooms_trace = go.Scattermap(
                    mode="markers",
                    lon=ROOM_CENTERS["lon"], lat=ROOM_CENTERS["lat"],
                    text=ROOM_CENTERS["name"], hoverinfo="text",
                    marker={'size': 4, 'color': 'rgba(0, 120, 255, 0.4)'},
                    name="Rooms"
                )

                # Trace 2: The actual navigation route
                route_trace = go.Scattermap(
                    mode="markers+lines",
                    lon=lons, lat=lats,
                    marker={'size': 12, 'color': '#FF0000'},
                    line={'width': 5, 'color': '#FF0000'},
                    text=texts, hoverinfo="text",
                    name="Route"
                )

                # Add BOTH traces to the figure
                fig = go.Figure(data=[rooms_trace, route_trace])

                fig.update_layout(
                    map=dict(
                        style="open-street-map",
                        center={'lon': lons[0], 'lat': lats[0]},
                        zoom=18.5,
                        # The background polygons remain exactly the same
                        layers=[
                            dict(
                                sourcetype='geojson',
                                source=ROOM_GEOJSON,
                                type='fill',
                                color='rgba(0, 120, 255, 0.15)'
                            ),
                            dict(
                                sourcetype='geojson',
                                source=ROOM_GEOJSON,
                                type='line',
                                color='blue',
                                line=dict(width=1)
                            )
                        ]
                    ),
                    margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
                    height=500,
                    showlegend=False
                )

                st.session_state.current_map = fig
                st.session_state.current_instructions = data.get("instructions", [])

            if data.get("target_id"):
                st.session_state.current_node_id = data["target_id"]

            st.session_state.chat_history.append({"role": "assistant", "content": agent_reply})
            with st.chat_message("assistant"):
                st.markdown(agent_reply)

            st.rerun()

        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")