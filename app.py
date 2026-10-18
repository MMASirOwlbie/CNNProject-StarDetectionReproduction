%%writefile /content/app.py
import streamlit as st

st.set_page_config(
    page_title="Star Detection",
    page_icon="🔭"
)

st.title("CNN-Based Star Detection")
st.write(
    "This application demonstrates our "
    "reproduced star detection model."
)

st.info("The model inference pipeline will be connected next.")
