import os
import pandas as pd
import numpy as np
import streamlit as st
import urllib.request
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Final Brain Strategy Engine", page_icon="🧠", layout="wide")
st.title("🧠 Final Brain: Master Strategy Engine")
st.markdown("Querying **102,451 records** across Academic Journals, Strategic Books, and Creative Case Studies.")

# YOUR PRODUCTION DATA LINKS
CSV_CLOUD_URL = "https://huggingface.co/spaces/jonasssssssssingelmann/final-brain-engine/resolve/main/marketing_intelligence_master.csv?download=true"
NPY_CLOUD_URL = "https://huggingface.co/spaces/jonasssssssssingelmann/final-brain-engine/resolve/main/marketing_intelligence_vectors.npy?download=true"

@st.cache_resource
def load_production_assets():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    csv_path = "marketing_intelligence_master.csv"
    vec_path = "marketing_intelligence_vectors.npy"
    
    # Download logic if the files are not present on the Streamlit server
    if not os.path.exists(csv_path):
        with st.spinner("📥 Downloading 102k database from cloud space (approx 140MB)..."):
            urllib.request.urlretrieve(CSV_CLOUD_URL, csv_path)
            
    if not os.path.exists(vec_path):
        with st.spinner("📥 Downloading vector coordinate matrix (approx 157MB)..."):
            urllib.request.urlretrieve(NPY_CLOUD_URL, vec_path)
            
    df = pd.read_csv(csv_path)
    vectors = np.load(vec_path)
    return model, df, vectors

try:
    with st.spinner("Loading strategy matrix into server memory..."):
        model, df_master, matrix_vectors = load_production_assets()
    st.success("✨ Strategy Matrix Active and Cached!")
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# ── USER INTERFACE INPUTS ───────────────────────────────────────────────────
st.sidebar.header("📋 Client Briefing Inputs")
client_name = st.sidebar.text_input("Client Name", "Powtoon Pilot")

available_domains = sorted(df_master['marketing_domain'].dropna().unique().tolist())
domain_filter = st.sidebar.selectbox("Filter by Marketing Domain", ["All Domains"] + available_domains)

core_tension = st.sidebar.text_area(
    "Core Consumer Tension / Problem", 
    "Users view presentation software as a boring, static utility tool rather than a dynamic storytelling asset."
)

target_behavior = st.sidebar.text_area(
    "Target Behavior / Objective", 
    "Triggering loss aversion and micro-habits during the 7-day trial onboarding window to drive self-serve premium conversions."
)

top_n = st.sidebar.slider("Number of Matches to Discover", 1, 5, 3)

# ── SEARCH EXECUTION ENGINE ─────────────────────────────────────────────────
if st.sidebar.button("🚀 Run Strategic Alignment Search"):
    st.subheader(f"💡 Top Alignments Discovered for: {client_name}")
    
    query_text = f"Framework: {core_tension} | Playbook Steps: {target_behavior}"
    
    with st.spinner("Calculating geometric cosine distances..."):
        query_vector = model.encode(query_text, convert_to_numpy=True)
        
        dot_product = np.dot(matrix_vectors, query_vector)
        query_norm = np.linalg.norm(query_vector)
        matrix_norms = np.linalg.norm(matrix_vectors, axis=1)
        matrix_norms[matrix_norms == 0] = 1e-9
        
        scores = dot_product / (query_norm * matrix_norms)
        
        df_results = df_master.copy()
        df_results['match_fidelity'] = scores
        
        if domain_filter != "All Domains":
            df_results = df_results[df_results['marketing_domain'] == domain_filter]
            
        discoveries = df_results.sort_values(by='match_fidelity', ascending=False).head(top_n)
        
    for idx, (_, row) in enumerate(discoveries.iterrows()):
        source_tag = "🎓 Science Layer" if row['master_source_type'].lower() == "journal" else "🏆 Practical Precedent"
        
        with st.container():
            st.markdown(f"### Match #{idx+1} | Score: `{row['match_fidelity']:.2f}` — **{source_tag}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Origin:** {row['file_origin']}")
                st.markdown(f"**Index Context:** {row['structural_index']}")
            with col2:
                st.markdown(f"**Core Framework:** {row['title_or_context']}")
                st.markdown(f"**Domain:** {row['marketing_domain']}")
                
            st.info(f"**Theoretical Framework / Creative Rule:**\n\n{row['core_insight_framework']}")
            st.success(f"**Actionable Execution Playbook Steps:**\n\n{row['actionable_playbook_steps']}")
            st.divider()
