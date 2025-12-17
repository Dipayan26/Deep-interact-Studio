# FRONTEND APP.py
#################################################

import os
import requests
import streamlit as st


pg = st.navigation([st.Page("home.py"), st.Page("build.py"), st.Page("job_status.py")])
pg.run()

# st.caption(f"Backend URL: {BACKEND_URL}")



