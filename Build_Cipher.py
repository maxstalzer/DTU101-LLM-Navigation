import json
import itertools
import math
from shapely.geometry import shape, Point
from shapely.ops import nearest_points

def escape_string(s):
    if not s: return ""
    return str(s).replace('"', '\\"').replace("'", "\\'")

def build_neo4j_cypher_seed():
    input_file = "data/dtu_semantic_nodes.json"
    output_file = "data/dtu_neo4j_seed.cypher"

    # --- Tiered distance thresholds by room type ---
    # Large/open rooms need a looser threshold to connect to corridors
    DISTANCE_THRESHOLDS = {
        ("room", "room"):         3.0,
        ("room", "corridor"):     6.0,
        ("corridor", "corridor"): 6.0,
        ("canteen", "corridor"):  8.0,
        ("canteen", "room"):      8.0,
        ("databar", "corridor"):  8.0,
        ("databar", "room"):      8.0,
        ("stairs", "room"):       6.0,
        ("elevator", "room"):     6.0,
        ("default", "default"):   5.0,  # fallback for any unmatched pair
    }

    def get_threshold(type_a, type_b):
        key = tuple(sorted([type_a, type_b]))
        return DISTANCE_THRESHOLDS.get(key, DISTANCE_THRESHOLDS[("default", "default")])

    print(f"🧠 Loading semantic nodes from {input_file}...")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            nodes_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {input_file} not found. Run extract_mazemap.py first.")
        return

    nodes_101 = {k: v for k, v in nodes_data.items() if '101' in str(v.get('building_name', ''))}
    print(f"🏢 Found {len(nodes_101)} nodes in Building 101.")

    cypher_statements = [
        "// ==========================================",
        "// DTU Building 101: Semantic Knowledge Graph",
        "// ==========================================\n",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Space) REQUIRE s.poi_id IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Portal) REQUIRE p.portal_id IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Building) REQUIRE b.building_id IS UNIQUE;",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Floor) REQUIRE f.floor_id IS UNIQUE;\n"
    ]

    parsed_spaces = []
    buildings_seen = set()
    floors_seen = set()
    polygon_fallback_count = 0

    print("🧩 Processing Geometries and writing Node statements...")
    for node_id, attr in nodes_101.items():
        x, y = attr.get('center_x'), attr.get('center_y')
        if x is None or y is None:
            continue

        poly = None
        geom = attr.get('polygon')
        if geom and geom.get('type') in ['Polygon', 'MultiPolygon']:
            try:
                poly = shape(geom)
                # Validate — degenerate polygons cause silent distance failures
                if not poly.is_valid or poly.area < 0.1:
                    poly = None
            except Exception:
                poly = None

        if poly is None:
            # Use a larger buffer (5m) so fallback nodes still connect
            poly = Point(x, y).buffer(5.0)
            polygon_fallback_count += 1

        floor_id = attr.get('floor_id') or f"floor_{attr.get('z_level')}"
        building_id = attr.get('building_id') or "bldg_101"
        maze_kind = attr.get('maze_kind', 'room').lower()

        parsed_spaces.append({
            'id': node_id,
            'floor_id': floor_id,
            'floor': attr.get('z_level'),
            'type': maze_kind,
            'poly': poly,
            'x': x, 'y': y,
            'has_real_polygon': geom is not None
        })

        if building_id not in buildings_seen:
            b_name = escape_string(attr.get('building_name', 'Building 101'))
            cypher_statements.append(f'MERGE (:Building {{building_id: "{building_id}", name: "{b_name}"}});')
            buildings_seen.add(building_id)

        if floor_id not in floors_seen:
            f_name = escape_string(attr.get('floor_name', str(attr.get('z_level'))))
            cypher_statements.append(f'MERGE (f:Floor {{floor_id: "{floor_id}", name: "{f_name}", level: {attr.get("z_level")}}});')
            cypher_statements.append(f'MATCH (f:Floor {{floor_id: "{floor_id}"}}), (b:Building {{building_id: "{building_id}"}}) MERGE (b)-[:HAS_STOREY]->(f);')
            floors_seen.add(floor_id)

        label = f"Space:{attr.get('node_label', 'Room')}"
        if attr.get('is_wheelchair_accessible'):
            label += ":Accessible"

        name = escape_string(attr.get('name'))
        desc = escape_string(attr.get('description'))

        stmt = (
            f'CREATE (:{label} {{'
            f'poi_id: "{node_id}", name: "{name}", description: "{desc}", '
            f'maze_kind: "{maze_kind}", x: {x}, y: {y}, floor: {attr.get("z_level")}'
            f'}});'
        )
        cypher_statements.append(stmt)
        cypher_statements.append(
            f'MATCH (s:Space {{poi_id: "{node_id}"}}), (f:Floor {{floor_id: "{floor_id}"}}) MERGE (f)-[:HAS_SPACE]->(s);'
        )

    print(f"⚠️  {polygon_fallback_count} nodes had no polygon — used 5m buffer fallback.")
    cypher_statements.append("\n// 2. Generate Adjacency Portals (Doors) and Walking Paths\n")

    floors = {}
    for space in parsed_spaces:
        floors.setdefault(str(space['floor']), []).append(space)

    added_portals = 0
    added_edges = 0

    print("🚪 Calculating Wall Intersections to inject Portals...")
    for floor_level, spaces in floors.items():
        for s1, s2 in itertools.combinations(spaces, 2):
            p1, p2 = s1['poly'], s2['poly']
            threshold = get_threshold(s1['type'], s2['type'])

            if p1.distance(p2) < threshold:
                np1, np2 = nearest_points(p1, p2)
                portal_x = (np1.x + np2.x) / 2.0
                portal_y = (np1.y + np2.y) / 2.0

                room_a, room_b = sorted([s1['id'], s2['id']])
                portal_id = f"portal_{room_a}_{room_b}"

                dist_a_to_p = math.hypot(s1['x'] - portal_x, s1['y'] - portal_y)
                dist_b_to_p = math.hypot(s2['x'] - portal_x, s2['y'] - portal_y)

                cypher_statements.append(
                    f'MERGE (:Portal {{portal_id: "{portal_id}", x: {portal_x}, y: {portal_y}, floor: {floor_level}}});'
                )
                # s1 → portal → s2
                cypher_statements.append(
                    f'MATCH (a:Space {{poi_id: "{s1["id"]}"}}), (p:Portal {{portal_id: "{portal_id}"}}) '
                    f'MERGE (a)-[:HAS_EXIT {{distance: {round(dist_a_to_p, 2)}}}]->(p);'
                )
                cypher_statements.append(
                    f'MATCH (p:Portal {{portal_id: "{portal_id}"}}), (b:Space {{poi_id: "{s2["id"]}"}}) '
                    f'MERGE (p)-[:CONNECTS_TO {{distance: {round(dist_b_to_p, 2)}}}]->(b);'
                )
                # s2 → portal → s1 (explicit reverse so shortestPath works in both directions)
                cypher_statements.append(
                    f'MATCH (b:Space {{poi_id: "{s2["id"]}"}}), (p:Portal {{portal_id: "{portal_id}"}}) '
                    f'MERGE (b)-[:HAS_EXIT {{distance: {round(dist_b_to_p, 2)}}}]->(p);'
                )
                cypher_statements.append(
                    f'MATCH (p:Portal {{portal_id: "{portal_id}"}}), (a:Space {{poi_id: "{s1["id"]}"}}) '
                    f'MERGE (p)-[:CONNECTS_TO {{distance: {round(dist_a_to_p, 2)}}}]->(a);'
                )

                added_portals += 1
                added_edges += 4

    cypher_statements.append("\n// 3. Generate Vertical Transitions (Stairs/Elevators)\n")

    print("🧗 Mapping vertical transitions...")
    transitions = [s for s in parsed_spaces if s['type'] in ['stairs', 'elevator']]
    for t1, t2 in itertools.combinations(transitions, 2):
        if t1['type'] == t2['type'] and abs(t1['floor'] - t2['floor']) == 1.0:
            dist = math.hypot(t1['x'] - t2['x'], t1['y'] - t2['y'])
            if dist < 15.0:
                rel_type = "CONNECTS_VIA_STAIRS" if t1['type'] == 'stairs' else "CONNECTS_VIA_ELEVATOR"
                cypher_statements.append(
                    f'MATCH (t1:Space {{poi_id: "{t1["id"]}"}}), (t2:Space {{poi_id: "{t2["id"]}"}}) '
                    f'MERGE (t1)-[:{rel_type} {{distance: 5.0}}]->(t2);'
                )
                cypher_statements.append(
                    f'MATCH (t1:Space {{poi_id: "{t1["id"]}"}}), (t2:Space {{poi_id: "{t2["id"]}"}}) '
                    f'MERGE (t2)-[:{rel_type} {{distance: 5.0}}]->(t1);'
                )
                added_edges += 2

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(cypher_statements))

    print(f"\n✅ GRAPH GENERATION COMPLETE")
    print(f"   Portals injected : {added_portals}")
    print(f"   Relationships    : {added_edges}")
    print(f"   Polygon fallbacks: {polygon_fallback_count}")

if __name__ == "__main__":
    build_neo4j_cypher_seed()