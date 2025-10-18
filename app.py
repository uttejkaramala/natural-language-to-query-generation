# app.py
import streamlit as st
import os
import sys  
import json
import logging
import pandas  as pd
import time
from dotenv import load_dotenv

# --- FIX FOR MODULE NOT FOUND ERRORS ---
# Add the project root directory to the Python path to resolve local imports (core_logic)
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 

# LangChain Imports
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.utilities import SQLDatabase
from sqlalchemy.exc import OperationalError
from sqlalchemy import create_engine
from pymongo.errors import ServerSelectionTimeoutError

# Local Logic Imports
from core_logic.db_connectors import create_sql_db_uri, initialize_sql_db, initialize_mongo_client, execute_mongo_query
from core_logic.nl2db_agent import create_sql_agent_executor, create_mongo_chain
from core_logic.util import create_download_filename, convert_to_csv
from core_logic.visualization import generate_chart

# --- Setup ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
st.set_page_config(layout="wide", page_title="Gemini NL2DB Query Engine")
logging.basicConfig(level=logging.INFO)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "connection_status" not in st.session_state:
    st.session_state["connection_status"] = "Not Connected"
if "db_agent" not in st.session_state:
    st.session_state["db_agent"] = None
if "db_type" not in st.session_state:
    st.session_state["db_type"] = None
if "mongo_chain_data" not in st.session_state:
    st.session_state["mongo_chain_data"] = None


# --- UI Components ---

def connect_db(db_params):
    """Handles database connection and agent creation."""
    db_type = db_params["db_type"]
    st.session_state["db_type"] = db_type
    
    if not GEMINI_API_KEY:
        st.error("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")
        return

    try:
        if db_type in ["MySQL", "PostgreSQL"]:
            # 1. Create URI
            db_uri = create_sql_db_uri(**db_params)
            
            # 2. Initialize LangChain SQLDatabase and Agent
            db = initialize_sql_db(db_uri)
            # store db so we can run raw SQL queries if the agent provides them
            st.session_state["sql_db"] = db
            # also store a SQLAlchemy engine (pandas expects an engine/connection)
            try:
                sql_engine = create_engine(db_uri)
                st.session_state["sql_engine"] = sql_engine
            except Exception:
                st.session_state["sql_engine"] = None
            
            # Display a simplified schema for verification
            st.sidebar.success(f"Connected to {db_type}. Introspecting schema...")
            # We use a placeholder handler since the main agent uses the full callback
            agent_callback = StreamlitCallbackHandler(st.empty()) 
            
            with st.spinner(f"Initializing LangChain {db_type} Agent..."):
                agent = create_sql_agent_executor(db, GEMINI_API_KEY, agent_callback)
                st.session_state["db_agent"] = agent
            
            st.session_state["connection_status"] = f"Connected to {db_type}"
            st.session_state["messages"] = [{"role": "assistant", "content": f"Successfully connected to **{db_type}** and initialized the Gemini Agent. Ask me a question about your data! (e.g., 'Show me the top 5 customers')."}]
            
        elif db_type == "MongoDB":
            # 1. Initialize PyMongo client
            client = initialize_mongo_client(db_params["host"], db_params["port"], db_params["user"], db_params["password"])
            client.admin.command('ping') # Test connection
            
            # 2. Create MongoDB Chain Data
            mongo_chain_data = create_mongo_chain(client, db_params["database"], GEMINI_API_KEY)
            st.session_state["mongo_chain_data"] = mongo_chain_data
            
            st.session_state["connection_status"] = f"Connected to {db_type}"
            st.session_state["messages"] = [{"role": "assistant", "content": f"Successfully connected to **{db_type}**. Ask me a question about the `{db_params['database']}` database! (e.g., 'Find the average price for products in the 'electronics' category')."}]
        
    except (OperationalError, ServerSelectionTimeoutError) as e:
        st.error(f"Connection Failed: {e}")
        st.session_state["connection_status"] = "Connection Failed"
        st.session_state["db_agent"] = None
    except Exception as e:
        st.exception(e)
        st.session_state["connection_status"] = "Error"
        st.session_state["db_agent"] = None
        
def run_mongo_query(user_question: str):
    """Executes the MongoDB chain for NoSQL databases."""
    mongo_data = st.session_state["mongo_chain_data"]
    llm = mongo_data["llm"]
    prompt_template = mongo_data["prompt_template"]
    client = mongo_data["client"]
    db_name = mongo_data["db_name"]
    schema_info = mongo_data["schema_info"]
    
    # 1. Generate MQL Pipeline (LLM call)
    full_prompt = prompt_template.format(schema_info=schema_info, user_question=user_question)
    
    with st.spinner("Generating MongoDB Query (MQL)..."):
        # We invoke the LLM to get the MQL pipeline list as a string
        mql_output = llm.invoke(full_prompt).content
        
    try:
        # Attempt to parse the MQL query string (which should be a raw Python list)
        mql_pipeline = json.loads(mql_output)
        if not isinstance(mql_pipeline, list):
            st.error(f"LLM did not return a valid MQL list. Output: {mql_output}")
            st.session_state.messages.append({"role": "assistant", "content": f"Query generation failed. LLM output: ```json\n{mql_output}\n```"})
            return

        # The LLM must also infer which collection to query, which is complex.
        # For simplicity, we assume the user/LLM focuses on the first collection in the schema.
        # A more robust solution would require the LLM to output the collection name as well (e.g., using Pydantic).
        collection_names = list(json.loads(schema_info).keys())
        target_collection = collection_names[0] if collection_names else "default_collection"

        # 2. Execute MQL Pipeline
        result = execute_mongo_query(client, db_name, target_collection, mql_pipeline)
        
        # 3. Format and display results directly as table (no textual pipeline dump)
        if isinstance(result, str) and result.startswith("Error"):
            st.session_state.messages.append({"role": "assistant", "content": result})
        else:
            # result is expected to be {"columns": [...], "rows": [...]}
            df = None
            try:
                import pandas as _pd
                df = _pd.DataFrame(result["rows"], columns=result["columns"])
            except Exception:
                try:
                    # If result is a list of dicts
                    df = _pd.DataFrame(result)
                except Exception:
                    df = None

            if df is not None:
                st.session_state.messages.append({"role": "data", "content": {"columns": list(df.columns), "rows": df.values.tolist()}})
                st.session_state.messages.append({"role": "assistant", "content": f"Query executed successfully. Displaying results from collection `{target_collection}`."})
                st.dataframe(df, width='stretch')
            else:
                st.session_state.messages.append({"role": "assistant", "content": f"Query executed. Raw result: {result}"})

    except json.JSONDecodeError:
        st.error("LLM did not return a valid MQL list (JSON decoding error).")
        st.session_state.messages.append({"role": "assistant", "content": f"Query generation failed. LLM output: ```\n{mql_output}\n```"})
    except Exception as e:
        st.exception(e)
        st.session_state.messages.append({"role": "assistant", "content": f"An unexpected error occurred: {e}"})


# --- Main Application Layout ---

st.title("💡 Gemini NL2DB Query Engine")
st.caption("Converse with your local MySQL, PostgreSQL, or MongoDB using natural language.")

# Sidebar for Configuration
with st.sidebar:
    st.header("Database Configuration")
    
    db_type = st.selectbox("Database Type", ["MySQL", "PostgreSQL", "MongoDB"])
    host = st.text_input("Host", "localhost")
    port = st.text_input("Port", "3306" if db_type == "MySQL" else ("5432" if db_type == "PostgreSQL" else "27017"))
    user = st.text_input("User", "root")
    password = st.text_input("Password", "password", type="password")
    database = st.text_input("Database Name/Schema", "testdb")
    
    connect_button = st.button("Connect & Initialize Agent")
    st.markdown(f"**Connection Status:** {st.session_state['connection_status']}")

    if connect_button:
        db_params = {
            "db_type": db_type,
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database
        }
        connect_db(db_params)

from datetime import datetime

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        if message["role"] == "data":
            result_data = message["content"]
            if "rows" in result_data and "columns" in result_data:
                df = pd.DataFrame(result_data["rows"], columns=result_data["columns"])
                if df.empty:
                    st.warning("No data returned from query.")
                    continue

                message_container = st.container()
                if generate_chart(df, message_container):
                    st.divider()

                st.dataframe(df, width='stretch')
                csv_data = convert_to_csv(df)

                user_qn = st.session_state.get("last_user_question", "query_result")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                file_name = f"{create_download_filename(user_qn).replace('.csv', '')}_{timestamp}.csv"

                unique_key = f"download_button_{int(time.time() * 1000)}"
                st.download_button(
                    label="Download Query Results as CSV",
                    data=csv_data,
                    file_name=file_name,
                    mime='text/csv',
                    key=unique_key
                )
            else:
                st.error("Invalid data format received.")
        else:
            st.markdown(message["content"])

# User Input Handling
if st.session_state["connection_status"].startswith("Connected"):
    user_prompt = st.chat_input("Ask a question about your database...")
    
    if user_prompt:
        # store the user's prompt for later use (filenames, context)
        st.session_state["last_user_question"] = user_prompt
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            if st.session_state["db_type"] in ["MySQL", "PostgreSQL"]:
                # --- SQL Agent Execution ---
                agent_executor = st.session_state["db_agent"]
                
                # Setup callback for displaying agent thoughts
                st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
                
                try:
                    response = agent_executor.invoke({"input": user_prompt}, {"callbacks": [st_callback]})

                    # Try to extract SQL query from intermediate steps (heuristic)
                    sql_query = None
                    try:
                        intermediate = response.get("intermediate_steps") or response.get("intermediate_steps", [])
                        for step in reversed(intermediate):
                            # step might be (action, observation) or dict
                            action = None
                            if isinstance(step, (list, tuple)) and len(step) >= 1:
                                action = step[0]
                            elif isinstance(step, dict):
                                action = step.get("action") or step

                            if action:
                                # action may be dict-like or object-like
                                tool_input = None
                                if isinstance(action, dict):
                                    tool_input = action.get("tool_input") or action.get("input") or action.get("code")
                                else:
                                    tool_input = getattr(action, "tool_input", None) or getattr(action, "input", None)

                                if tool_input and ("select" in str(tool_input).lower() or "from" in str(tool_input).lower()):
                                    sql_query = str(tool_input)
                                    break
                    except Exception:
                        sql_query = None

                    # If SQL found, try to execute and display table
                    if sql_query:
                        try:
                            db_obj = st.session_state.get("sql_db")
                            import pandas as _pd
                            df = None
                            # Prefer the explicit SQLAlchemy engine we created at connect time
                            engine = st.session_state.get("sql_engine") or getattr(db_obj, "engine", None) or db_obj
                            try:
                                df = _pd.read_sql_query(sql_query, engine)
                            except Exception:
                                # fallback: try db_obj.run (LangChain SQLDatabase)
                                try:
                                    rows = db_obj.run(sql_query)
                                    # If rows is list of dicts convert to DataFrame
                                    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                                        df = _pd.DataFrame(rows)
                                except Exception:
                                    df = None

                            if df is not None:
                                st.session_state["messages"].append({"role": "data", "content": {"columns": list(df.columns), "rows": df.values.tolist()}})
                                st.dataframe(df, width='stretch')
                            else:
                                final_answer = response.get("output", "Could not retrieve an answer.")
                                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                                st.markdown(final_answer)
                        except Exception as e:
                            st.error(f"Error executing SQL: {e}")
                            final_answer = response.get("output", "Could not retrieve an answer.")
                            st.session_state.messages.append({"role": "assistant", "content": final_answer})
                            st.markdown(final_answer)
                    else:
                        # No SQL found; fallback to assistant NL answer
                        final_answer = response.get("output", "Could not retrieve an answer.")
                        st.session_state.messages.append({"role": "assistant", "content": final_answer})
                        st.markdown(final_answer)
                    
                    # NOTE: LangChain's SQL agent returns the final answer in NL. 
                    # For displaying the raw data, you'd typically extract the final SQL query 
                    # from the agent's thought process and run it manually, but for a conversational 
                    # flow, the NL summary is often preferred.
                    
                except Exception as e:
                    st.error(f"Agent Execution Error: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"An error occurred during agent execution: {e}"})

            elif st.session_state["db_type"] == "MongoDB":
                # --- MongoDB Chain Execution (Custom) ---
                # run the chain and get results as a structured dict
                run_mongo_query(user_prompt)

else:
    st.warning("Please configure your database connection and click 'Connect & Initialize Agent' to begin.")