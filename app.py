import streamlit as st
from streamlit_page import show_korean_page
from streamlit_page_en import show_english_page

# Page basic settings
st.set_page_config(layout="wide", page_title="Trading Strategy Generator")

# Initialize session state
if 'strategies' not in st.session_state:
    st.session_state['strategies'] = {}
if 'backtest' not in st.session_state:
    st.session_state['backtest'] = None

with st.sidebar:
    st.header("Language")
    lang = st.radio("Select Language", ["English", "한국어"])
    st.markdown("---")

if lang == "English":
    show_english_page()
else:
    show_korean_page()
