import streamlit as st

pg = st.navigation([st.Page("streamlit_proj.py", title='Mudel', icon=":material/calculate:"),
                    st.Page("meetod.py", title='Kirjeldus', icon=":material/tactic:"), 
                    st.Page("allikad.py", title='Andmeallikad', icon=":material/link:")])
pg.run()
