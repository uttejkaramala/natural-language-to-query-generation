import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import List, Dict, Any # Added Any for the type hint fix

# Threshold for unique values to consider a column categorical for grouping
MAX_CATEGORIES = 15
# Threshold for number of rows to decide if simple bar chart is appropriate
MIN_ROWS_FOR_PLOT = 5

def generate_chart(df: pd.DataFrame, message_container: Any) -> bool:
    """
    Analyzes the DataFrame and attempts to generate a simple chart if the data 
    appears suitable for a basic bar chart visualization.
    
    Args:
        df: The Pandas DataFrame containing the query results.
        message_container: The Streamlit container (e.g., st.chat_message) 
                           where the chart should be rendered.

    Returns:
        True if a chart was successfully generated, False otherwise.
    """
    if df.empty or len(df) < MIN_ROWS_FOR_PLOT:
        # Data is too sparse or empty for visualization
        return False

    # Identify potential categorical and numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Simple heuristic for potential grouping: one numerical column and one categorical
    potential_grouping_cols = [
        col for col in df.columns 
        if df[col].nunique() <= MAX_CATEGORIES and col not in numerical_cols
    ]

    if len(numerical_cols) >= 1 and len(potential_grouping_cols) >= 1:
        # Found a promising pair: use the first one from each category
        category_col = potential_grouping_cols[0]
        value_col = numerical_cols[0]

        try:
            # 1. Group by the categorical column and aggregate the numerical column
            grouped_data = df.groupby(category_col)[value_col].sum().reset_index()
            
            # 2. Check if the resulting grouped data is useful
            if len(grouped_data) > 1 and not grouped_data.isnull().values.any():
                
                with message_container:
                    st.subheader("Visualized Data")
                    st.caption(f"Showing total `{value_col}` grouped by `{category_col}`.")

                    # Use st.bar_chart for a fast, native Streamlit visualization
                    st.bar_chart(
                        data=grouped_data.set_index(category_col), 
                        width='stretch'
                    )
                return True
            
        except Exception as e:
            # Catch errors during grouping/plotting, e.g., type mismatch
            # st.warning(f"Failed to generate chart: {e}") 
            return False
            
    return False