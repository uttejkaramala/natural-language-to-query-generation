# core_logic/util.py

import pandas as pd
import io
import streamlit as st

# Caching this function is highly recommended by Streamlit documentation
# to prevent recomputing the file content on every app rerun.
# Note: st.cache_data is the modern decorator for data transformations.

@st.cache_data
def convert_to_csv(df: pd.DataFrame) -> bytes:
    """
    Converts a Pandas DataFrame into a CSV format encoded in UTF-8 bytes.
    This uses an in-memory buffer, avoiding temporary file creation on disk.
    
    Args:
        df: The Pandas DataFrame containing the query results.

    Returns:
        The CSV content as bytes, ready for st.download_button.
    """
    # Use StringIO to create an in-memory text buffer
    csv_buffer = io.StringIO()
    
    # Write the DataFrame to the buffer as CSV, excluding the index
    df.to_csv(csv_buffer, index=False)
    
    # Get the string value from the buffer and encode it to UTF-8 bytes
    return csv_buffer.getvalue().encode('utf-8')


def create_download_filename(user_question: str) -> str:
    """
    Creates a clean, short, and safe filename based on the user's question.
    """
    # 1. Truncate the question to max 30 characters
    clean_name = user_question[:30]
    
    # 2. Replace non-alphanumeric characters (excluding spaces/hyphens) with nothing
    clean_name = ''.join(c for c in clean_name if c.isalnum() or c in (' ', '-')).rstrip()
    
    # 3. Replace spaces and hyphens with underscores
    clean_name = clean_name.replace(' ', '_').replace('-', '_')
    
    # 4. Append the file extension
    return f"{clean_name}_result.csv"