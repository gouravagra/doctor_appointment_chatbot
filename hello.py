import streamlit as st
import time

# Streamlit App
st.title("Database Connection Test")
st.write("Testing database connection...")

# Keep the app alive
while True:
    st.success("App is running! 🚀")
    time.sleep(1)
