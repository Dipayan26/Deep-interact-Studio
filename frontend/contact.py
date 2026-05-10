import streamlit as st

st.title("Contact Us")
st.divider()

# ── Address ───────────────────────────────────────────────────────────────────
col_addr, col_hours = st.columns([3, 2])

with col_addr:
    st.subheader("Address")
    st.markdown("""
**Computational Systems Biology Laboratory**

Department of Bioinformatics
University of North Bengal
Raja Rammohunpur, P.O. - N.B.U.
District - Darjeeling, PIN - 734013
West Bengal, India
""")
    st.link_button(
        "Open in Google Maps",
        "https://maps.app.goo.gl/nY1QXVG4jWnZimCQ7",
        icon=":material/map:",
    )

with col_hours:
    st.subheader("Office Hours")
    st.markdown("""
**Monday – Friday**
10:00 AM – 5:00 PM IST
""")
    st.link_button(
        "Lab Website",
        "https://compbiosysnbu.in",
        icon=":material/open_in_new:",
    )

st.divider()

# ── Principal Investigator ────────────────────────────────────────────────────
st.subheader("Principal Investigator")

with st.container(border=True):
    st.markdown("""
**Dr. Chiranjib Sarkar**

Assistant Professor

Computational Systems Biology Lab · Department of Bioinformatics · University of North Bengal
""")
    st.markdown(
        "[:material/email: chiranjib@nbu.ac.in](mailto:chiranjib@nbu.ac.in)"
    )

st.divider()

# ── Developers ────────────────────────────────────────────────────────────────
st.subheader("Developers")

dev1_col, dev2_col, dev3_col = st.columns(3)

with dev1_col:
    with st.container(border=True):
        st.markdown("""
**Dipayan Sarkar**

Senior Research Fellow (UGC-CSIR-SRF)

Computational Systems Biology Lab
Department of Bioinformatics
University of North Bengal
""")
        st.markdown(
            "[:material/email: dipayansarkar26@gmail.com](mailto:dipayansarkar26@gmail.com)"
        )
        st.link_button(
            ":material/open_in_new:",
            "https://dipayansarkar.com/",
            help="Portfolio",
        )

with dev2_col:
    with st.container(border=True):
        st.markdown("""
**Koushik Bardhan**

Research Scholar

Computational Systems Biology Lab
Department of Bioinformatics
University of North Bengal
""")
        st.markdown(
            "[:material/email: koushikbardhan2000@gmail.com](mailto:koushikbardhan2000@gmail.com)"
        )
        st.link_button(
            ":material/open_in_new:",
            "https://koushikbardhan2000.github.io/portfolio/",
            help="Portfolio",
        )

with dev3_col:
    with st.container(border=True):
        st.markdown("""
**Bhuwan Sharma**

Research Scholar

Computational Systems Biology Lab
Department of Bioinformatics
University of North Bengal
""")
        st.markdown(
            "[:material/email: bhuwangautam09@gmail.com](mailto:bhuwangautam09@gmail.com)"
        )
        st.link_button(
            ":material/open_in_new:",
            "https://bhuwanshar.github.io/Portfolio/",
            help="Portfolio",
        )
