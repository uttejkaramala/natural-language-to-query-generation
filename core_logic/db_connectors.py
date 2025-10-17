# core_logic/db_connectors.py
import json
import logging
from typing import Dict, Any, Union

from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO)

# --- SQL Connection Helpers ---

def create_sql_db_uri(db_type: str, host: str, port: str, user: str, password: str, database: str) -> str:
    """Generates the SQLAlchemy connection URI."""
    if db_type == "MySQL":
        # Note: mysql+mysqlconnector is usually preferred over pymysql
        driver = "mysql+mysqlconnector" 
    elif db_type == "PostgreSQL":
        driver = "postgresql+psycopg2"
    else:
        raise ValueError(f"Unsupported SQL DB type: {db_type}")

    # Use a secure connection format
    return f"{driver}://{user}:{password}@{host}:{port}/{database}"

def initialize_sql_db(db_uri: str) -> SQLDatabase:
    """Initializes a LangChain SQLDatabase object."""
    engine = create_engine(db_uri)
    return SQLDatabase(engine)

# --- MongoDB Connection and Schema Helpers ---

def initialize_mongo_client(host: str, port: str, user: str, password: str) -> MongoClient:
    """Initializes a PyMongo client."""
    # MongoDB URI for local/self-managed connection
    if user and password:
        mongo_uri = f"mongodb://{user}:{password}@{host}:{port}/"
    else:
        mongo_uri = f"mongodb://{host}:{port}/"
    
    # Set a timeout for connection test
    return MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)

def extract_mongo_schema(client: MongoClient, db_name: str, max_samples: int = 3) -> str:
    """Extracts a sampled schema for all collections in a MongoDB database."""
    try:
        db = client.get_database(db_name)
        collection_names = db.list_collection_names()
        schema_info = {}

        for col_name in collection_names:
            collection = db[col_name]
            
            # Sample documents to infer structure
            sample_docs = list(collection.aggregate([
                {"$sample": {"size": max_samples}}
            ]))
            
            # Create a simple field structure
            if sample_docs:
                fields = {}
                for doc in sample_docs:
                    for key, value in doc.items():
                        if key not in fields:
                            # Extract type name (e.g., 'str', 'int', 'dict')
                            type_name = str(type(value)).split("'")[1]
                            fields[key] = type_name
                            
                            if isinstance(value, dict):
                                fields[key] = f"Object with keys: {list(value.keys())}"
                            
                            # CORRECTED LINE for array handling
                            elif isinstance(value, list):
                                # Determine element type for better LLM context
                                element_type = str(type(value[0])).split("'")[1] if value else 'mixed'
                                fields[key] = f"Array of {element_type}"

                schema_info[col_name] = {
                    "sample_count": len(sample_docs),
                    "fields": fields
                }
            else:
                schema_info[col_name] = "No documents in this collection."

        return json.dumps(schema_info, indent=2)

    except Exception as e:
        logging.error(f"Error extracting MongoDB schema: {e}")
        return f"Error: Could not extract schema. {e}"

# --- Execution Helper ---
def execute_mongo_query(client: MongoClient, db_name: str, collection_name: str, pipeline: list) -> Union[Dict[str, Any], str]:
    """Executes a MongoDB aggregation pipeline and returns results."""
    try:
        db = client.get_database(db_name)
        collection = db[collection_name]
        
        # Check for non-read operations
        if any(key in str(pipeline).lower() for key in ["$insert", "$delete", "$update", "$merge"]):
            return "Error: Non-read operations (insert, delete, update, merge) are forbidden."
            
        results = list(collection.aggregate(pipeline))
        
        # Format results (assuming results are flat or simple for display)
        if results:
            columns = list(results[0].keys())
            rows = [list(row.values()) for row in results]
            
            return {
                "columns": columns,
                "rows": rows
            }
        
        return {"columns": ["Result"], "rows": [["No results returned."]]}

    except Exception as e:
        return f"Error during query execution: {e}"
