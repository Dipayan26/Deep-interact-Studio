import streamlit as st

pg = st.navigation({
    "": [
        st.Page("home.py",       title="Home",       icon=":material/home:"),
        st.Page("manual.py",     title="Manual",     icon=":material/menu_book:"),
        st.Page("contact.py",    title="Contact Us", icon=":material/contact_mail:"),
        st.Page("references.py", title="References", icon=":material/library_books:"),
    ],
    "Model Building": [
        st.Page("ppi.py",              title="PPI Prediction",           icon=":material/hub:"),
        st.Page("dti.py",              title="Drug–Target Interaction",   icon=":material/medication:"),
        st.Page("rna_prot.py",         title="RNA–Protein Interaction",   icon=":material/genetics:"),
        st.Page("prot_dna.py",         title="Protein–DNA Interaction",   icon=":material/biotech:"),
    ],
    "Tools": [
        st.Page("inference.py",     title="Inference",      icon=":material/play_arrow:"),
        st.Page("check_results.py", title="Check Results",  icon=":material/monitor_heart:"),
        st.Page("job_status.py",    title="Job Status",     icon=":material/list_alt:"),
    ],
})
pg.run()
