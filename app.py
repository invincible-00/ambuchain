"""
AmbuChain - Streamlit entry point.

Run from the project root:   streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="AmbuChain · Workflow Analysis", page_icon="🚑", layout="wide",
                   initial_sidebar_state="expanded")

from frontend import dashboard  # noqa: E402  (after set_page_config)

dashboard.run()
