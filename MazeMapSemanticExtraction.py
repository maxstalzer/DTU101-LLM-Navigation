import requests
import json
import time

def fetch_dtu_mazemap_nodes():
    campus_id = 89  
    print(f"Fetching spatial nodes for DTU Lyngby (Campus {campus_id})...")
    
    headers = {
        "User-Agent": "DTU-Student-Accessibility-Project/2.0-Neo4j"
    }
    
    kg_nodes = {}
    limit = 2000  
    offset = 0
    total_raw_pois = 0
    
    seen_poi_ids = set() 
    
    while True:
        poi_url = f"https://api.mazemap.com/api/pois/?campusid={campus_id}&limit={limit}&offset={offset}"
        response = requests.get(poi_url, headers=headers)
        
        if response.status_code != 200:
            print(f"Error fetching data. HTTP Status: {response.status_code}")
            break
            
        data = response.json()
        pois = data.get("pois", [])
        
        if not pois:
            break
            
        total_raw_pois += len(pois)
        print(f"📥 Downloaded batch of {len(pois)} POIs... (Total raw: {total_raw_pois})")
        
        new_nodes_found = 0
        
        for poi in pois:
            poi_id = poi.get("poiId")
            
            # Skip duplicates and deleted/inactive POIs
            if poi_id in seen_poi_ids:
                continue
            if poi.get("deleted", False) or not poi.get("active", True):
                continue
                
            seen_poi_ids.add(poi_id)
            new_nodes_found += 1
            
            title = poi.get("title")
            if not title: continue 
                
            # Get Spatial Coordinates
            point_data = poi.get("point", {})
            coords = point_data.get("coordinates", [])
            if not coords or len(coords) < 2: continue
            x, y = coords[0], coords[1]
                
            # Get the 2D Polygon for Spatial Adjacency Reasoning (Crucial for Neo4j Portals)
            polygon = poi.get("geometry", {})

            # Get Semantic Data
            z_level = poi.get("z", 0.0)
            identifier = poi.get("identifier", "") # Formal room code (e.g., 101-2.910A)
            description = poi.get("description", "")
            categories = poi.get("categories", [])
            types = poi.get("types", [])
            
            title_lower = title.lower()
            desc_lower = str(description).lower()
            kind = poi.get("kind", "room").lower()
            
            # --- NEO4J NODE CLASSIFICATION HEURISTICS ---
            is_stairs = "stair" in title_lower or "trappe" in title_lower
            is_elevator = "elevator" in title_lower or "lift" in title_lower
            is_restroom = "toilet" in title_lower or "restroom" in title_lower or "wc" in title_lower.split()
            is_accessible_restroom = is_restroom and ("hc" in title_lower or "handicap" in title_lower or "disabled" in title_lower)
            is_canteen = "kantine" in title_lower or "canteen" in title_lower or "cafe" in title_lower
            
            # Determine the primary Neo4j Label for this node
            node_type = "stairs" if is_stairs else \
                        "elevator" if is_elevator else \
                        "accessible_restroom" if is_accessible_restroom else \
                        "restroom" if is_restroom else \
                        "canteen" if is_canteen else kind
                
            # Build the Graph Node Dictionary
            kg_nodes[poi_id] = {
                # Identity & Semantics (Neo4j Properties & Labels)
                "poi_id": poi_id,
                "identifier": identifier,
                "name": title,
                "description": description,
                "node_label": node_type.capitalize(), # e.g., 'Elevator', 'Canteen', 'Room'
                "maze_kind": kind,
                "maze_categories": categories,
                "maze_types": types,
                
                # Hierarchy Context (For (Building)-[:HAS_STOREY]->(Floor) relationships)
                "campus_id": campus_id,
                "building_id": poi.get("buildingId"),
                "building_name": poi.get("buildingName", "Unknown"),
                "floor_id": poi.get("floorId"),
                "floor_name": poi.get("floorName", str(z_level)),
                "z_level": z_level,
                
                # Space Syntax Geometry
                "center_x": x,  
                "center_y": y, 
                "polygon": polygon, 
                
                # Accessibility Ontology Flags
                "is_wheelchair_accessible": not is_stairs,
                "is_restroom": is_restroom,
                "is_accessible_restroom": is_accessible_restroom
            }
            
        if new_nodes_found == 0:
            print("⚠️ API is looping duplicates. Breaking pagination!")
            break
            
        if len(pois) < limit:
            break
            
        if total_raw_pois >= 10000:
            print("🛑 Hit safety cap of 10,000. Stopping to prevent runaway requests.")
            break
            
        offset += limit
        time.sleep(0.5) 
        
    output_filename = "data/dtu_semantic_nodes.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(kg_nodes, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Finished! Cleaned and saved {len(kg_nodes)} Neo4j-ready Nodes to '{output_filename}'!")

if __name__ == "__main__":
    fetch_dtu_mazemap_nodes()