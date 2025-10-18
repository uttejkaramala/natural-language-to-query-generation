# core_logic/nl2db_agent.py

from typing import Dict, Any, Union

from langchain.agents import AgentExecutor
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

from core_logic.db_connectors import extract_mongo_schema
from prompts.system_prompts import SQL_AGENT_SYSTEM_PROMPT, MONGO_CHAIN_PROMPT

# --- SQL AGENT FACTORY ---

def create_sql_agent_executor(db: SQLDatabase, llm_api_key: str, callback: StreamlitCallbackHandler) -> AgentExecutor:
    """Creates a LangChain SQL Agent Executor using Google Gemini."""
    
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=llm_api_key, 
        temperature=0, 
        # Streaming is useful but we rely on the callback for display
    )

    # Use a sliding window memory for the conversation history
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", 
        k=5, 
        return_messages=True,
        output_key="output"
    )

    # Custom prompt to guide the agent
    # We pass the custom prompt as the prefix
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=8,
        # Pass memory and callback to the executor
        agent_executor_kwargs={"memory": memory,
                               "return_intermediate_steps": True},
        callbacks=[callback],
        prefix=SQL_AGENT_SYSTEM_PROMPT
    )
    
    return agent_executor

# --- MQL CHAIN (For MongoDB) ---

def create_mongo_chain(client, db_name: str, llm_api_key: str) -> Dict[str, Any]:
    """Creates a LangChain-based pipeline for MongoDB (Custom MQL Chain)."""
    
    # 1. Get the schema
    schema_info = extract_mongo_schema(client, db_name)
    
    # 2. Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=llm_api_key, 
        temperature=0.0
    )

    # 3. Create the prompt template
    prompt_template = PromptTemplate.from_template(
        MONGO_CHAIN_PROMPT.format(schema_info="{schema_info}", user_question="{user_question}")
    )
    
    # The chain logic is simpler: NL -> Prompt -> LLM (MQL)
    # We return the LLM and schema to be used in the main Streamlit app.
    return {
        "llm": llm,
        "prompt_template": prompt_template,
        "schema_info": schema_info,
        "client": client,
        "db_name": db_name
    }
