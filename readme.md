💡 Gemini NL2DB Query Engine (Streamlit/LangChain)

This project allows users to query their local MySQL, PostgreSQL, or MongoDB databases using natural English language, leveraging Google's Gemini model via the LangChain framework.

The application is built entirely in Python using Streamlit for the interface.

🚀 Setup and Installation

1. Prerequisites

You need Python 3.9+ installed and access to a local database instance (MySQL, PostgreSQL, or MongoDB).

2. Get Your Gemini API Key

Obtain your API key from Google AI Studio and set it as an environment variable.

3. Clone and Setup Environment

# 1. Create the project directory (if starting fresh)

mkdir NL2DB-QueryBot
cd NL2DB-QueryBot

# 2. Save the files generated above into the correct structure:

# - Save app.py in the root.

# - Create core_logic/ and prompts/ directories and save files there.

mkdir core_logic prompts

# 3. Create and activate a virtual environment (Recommended)

python3 -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`

# 4. Install dependencies

pip install -r requirements.txt

4. Configure Environment Variables

Create a file named .env in the root directory and add your API key:

# .env file

GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

5. Running the Application

Execute the Streamlit application from the root directory:

streamlit run app.py

This will open the application in your web browser.

6. How to Use

Configure: In the left sidebar, select your Database Type and enter the connection details (Host, Port, User, Password, Database Name).

Connect: Click "Connect & Initialize Agent".

The app will attempt to connect, use SQLAlchemy or PyMongo to introspect the schema, and create the optimized LangChain agent (SQL Agent for SQL databases, or a custom LLM Chain for MongoDB).

Chat: Once connected, type your question in the chat box (e.g., "What are the names of the customers in the 'USA' and how many orders do they have?") and press Enter.

Results: The Gemini Agent will analyze the question, generate a SQL/MQL query, execute it against your live database, and provide a final natural language answer.
