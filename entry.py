# © 2026 Martin Klesment. Licensed under CC BY-NC 4.0.
import streamlit as st

pg = st.navigation([st.Page("streamlit_proj.py",    title='Mudel',        icon=":material/calculate:"),
                    st.Page("streamlit_proj_en.py", title='Model (EN)',    icon=":material/calculate:"),
                    st.Page("meetod.py",            title='Kirjeldus (Description)', icon=":material/tactic:")])
pg.run()
