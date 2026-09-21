"""Streamlit-specific resource caching kept outside model code."""
import streamlit as st
from src.emotion.model_loader import load_model


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()
