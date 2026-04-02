import streamlit as st
from main import PRISMRAG          # fix: class is in main.py, not prism_rag
from config import MAX_QUERY_LEN
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner UI

st.set_page_config(page_title="PRISM-RAG • Secure", layout="wide")
st.title("PRISM-RAG — Secure Anti-Lost-in-the-Middle Pipeline")

# ── Keep PRISMRAG instance alive across reruns ──────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = PRISMRAG()

rag = st.session_state.rag

# ── One-time ingestion ───────────────────────────────────────────────────────
if "ingested" not in st.session_state:
    with st.spinner("Ingesting documents securely (one-time)..."):
        rag.ingest()
    st.session_state.ingested = True
    st.success("✅ Documents ingested securely.")

# ── Main query ───────────────────────────────────────────────────────────────
query = st.text_input("Ask a question about your documents", max_chars=MAX_QUERY_LEN)

if st.button("Run Secure PRISM Pipeline"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running secure 5-layer pipeline..."):
            result = rag.run(query)

        st.markdown("### Answer")
        st.write(result["answer"])

        st.markdown("### Sources")
        for i, src in enumerate(result["sources"], 1):
            st.write(f"[{i}] {src}")

# ── Sidebar: Proof Tools ─────────────────────────────────────────────────────
st.sidebar.header("Proof & Demo")

demo_query = st.sidebar.text_input(
    "Failure Demo Query", "What is lost in the middle?"
)
if st.sidebar.button("Run Position Failure Demo"):
    with st.sidebar:
        with st.spinner("Running demo..."):
            demo = rag.run_position_failure_demo(demo_query)
        st.json(demo)

if st.sidebar.button("Run Ablation Study"):
    with st.sidebar:
        with st.spinner("Running ablation study (this takes a while)..."):
            try:
                results = rag.ablation_study()
                st.success("Ablation Results")
                st.table(results)
            except Exception as e:
                st.error(f"Error: {e}")