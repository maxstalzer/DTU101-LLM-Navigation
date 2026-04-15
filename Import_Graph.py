import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load credentials from your .env file
load_dotenv()
URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "Xamalam90") # Ensure this matches your local Neo4j password!

print("🚀 Connecting to Neo4j...")
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

try:
    driver.verify_connectivity()
    print("✅ Connection successful!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    exit()

print("📂 Loading Cypher file...")
with open("data/dtu_neo4j_seed.cypher", "r", encoding="utf-8") as f:
    raw_content = f.read()

# Split the massive file into individual queries using the semicolons
raw_queries = raw_content.split(";")
queries = []

for q in raw_queries:
    clean_q = q.strip()
    # Ignore empty strings or lines that are just comments
    if clean_q and not clean_q.startswith("//"):
        queries.append(clean_q)

print(f"⚙️ Executing {len(queries)} Cypher statements. Stand by...")

with driver.session() as session:
    for i, query in enumerate(queries):
        try:
            session.run(query)
            # Print an update every 500 statements so you know it's not frozen
            if (i + 1) % 500 == 0:
                print(f" ⏳ Processed {i + 1}/{len(queries)} statements...")
        except Exception as e:
            print(f"⚠️ Error on statement {i}: {e}")
            print(f"Query: {query[:100]}...") # Print a snippet of the failing query

print("🎉 Graph fully imported into Neo4j!")
driver.close()