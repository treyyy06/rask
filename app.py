import streamlit as st
import os
import tempfile
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

# Add workspace directory to path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from src.config import config
from src.logging_config import setup_logging
from src.answering.qa_pipeline import MultimodalQAPipeline

# Initialize logger
setup_logging()

# Set page configuration with a premium look
st.set_page_config(
    page_title="AQU-MR - Multimodal Document QA Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom premium CSS styles (harmonious dark mode palette, shadows, glassmorphism)
st.markdown("""
<style>
    /* Main body background styling */
    .stApp {
        background-color: #0d0e15;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Premium Header styling */
    .header-container {
        background: linear-gradient(135deg, #12102e 0%, #2b0f4a 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .header-title {
        font-size: 2.6rem !important;
        font-weight: 800;
        background: linear-gradient(to right, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card containers */
    .result-card {
        background: rgba(22, 28, 45, 0.6);
        border-radius: 12px;
        padding: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
    }
    
    /* Result headings */
    .result-heading {
        color: #f472b6 !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        text-transform: uppercase;
    }
    
    .badge-high { background-color: #064e3b; color: #34d399; border: 1px solid #059669; }
    .badge-medium { background-color: #78350f; color: #fbbf24; border: 1px solid #d97706; }
    .badge-low { background-color: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }
    
    /* Source pages list */
    .source-page-bubble {
        display: inline-block;
        width: 32px;
        height: 32px;
        line-height: 30px;
        border-radius: 50%;
        text-align: center;
        background-color: #1e1b4b;
        color: #e0e7ff;
        border: 1px solid #4f46e5;
        font-weight: 700;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# App Header Hero
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🔮 AQU-MR Pipeline</h1>
    <p class="header-subtitle">Adaptive Query Understanding & Modality-Aware Retrieval Engine</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Settings
st.sidebar.markdown("### 🛠️ Hardware & Controls")
api_key = st.sidebar.text_input("Gemini API Key", value=config.GEMINI_API_KEY, type="password")
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
    config.GEMINI_API_KEY = api_key

st.sidebar.markdown(f"""
* **Detected Processor:** Intel i5 CPU
* **System RAM:** 16 GB
* **Active GPU:** Intel Iris Xe
* **Mode:** CPU execution (offline fallbacks enabled)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload DocVQA PDF", type=["pdf"])

# Keep pipeline cached in session state to avoid re-running ingestion
if "pipeline" not in st.session_state:
    st.session_state.pipeline = MultimodalQAPipeline()
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

if uploaded_file is not None:
    if st.session_state.current_file_name != uploaded_file.name:
        with st.spinner("Processing document: rendering pages, extracting layout blocks, and building FAISS index..."):
            # Save uploaded file to raw data directory
            raw_path = config.RAW_DIR / uploaded_file.name
            with open(raw_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            try:
                st.session_state.pipeline.process_raw_documents(force=True)
                st.session_state.current_file_name = uploaded_file.name
                st.sidebar.success(f"Successfully Indexed: {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"Failed to process document: {e}")
                st.session_state.current_file_name = None
else:
    st.session_state.current_file_name = None
    st.info("👈 Please upload a PDF document in the sidebar to begin.")

# Main QA interface
if st.session_state.current_file_name is not None:
    st.markdown("### 💬 Ask a Modality-Aware Question")
    query = st.text_input("Enter your question:", placeholder="e.g. What was the GAAP Operating Income in 2024?")
    
    if query:
        with st.spinner("Executing AQU Query Engine & retrieval pools..."):
            result = st.session_state.pipeline.run_qa(query, debug=True)
            
        # Display Result
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<p class="result-heading">Grounded Answer</p>', unsafe_allow_html=True)
        st.write(result["answer"])
        
        # If it's a table answer, show formatted markdown
        if "table" in result["modalities"] and "|" in result["answer"]:
            st.markdown("#### Tabular Format Preview:")
            # Render Markdown Table cleanly
            st.markdown(result["answer"])

        # Format Badge for confidence
        conf = result["confidence"]
        badge_class = "badge-low"
        if conf == "HIGH":
            badge_class = "badge-high"
        elif conf == "MEDIUM":
            badge_class = "badge-medium"
            
        st.markdown(f"""
        <div style="margin-top: 1.5rem; display: flex; align-items: center; gap: 1rem;">
            <div>
                <strong>CONFIDENCE:</strong> 
                <span class="badge {badge_class}">{conf}</span>
            </div>
            <div>
                <strong>VALIDATION:</strong> 
                <span style="color: #f472b6; font-weight: 600;">{result['validation'].upper()}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Source pages list
        if result["pages"]:
            st.markdown("#### 📖 Source Pages & Evidence Provenance")
            cols_prov = st.columns(len(result["pages"]) + 1)
            with cols_prov[0]:
                st.write("**Modalities:**")
                st.write(f"- Text: {'✅' if result['evidence']['text'] else '❌'}")
                st.write(f"- Table: {'✅' if result['evidence']['table'] else '❌'}")
                st.write(f"- Image/Figure: {'✅' if result['evidence']['image'] else '❌'}")
                
            for idx, page_num in enumerate(result["pages"]):
                with cols_prov[idx + 1]:
                    st.markdown(f'<div style="text-align: center;"><div class="source-page-bubble">{page_num}</div><p style="font-size:0.8rem; margin-top:0.3rem;">Page {page_num}</p></div>', unsafe_allow_html=True)
                    # Display preview image if available
                    doc_id = result["doc_id"]
                    img_path = config.RENDERED_DIR / doc_id / f"page_{page_num}.png"
                    if img_path.exists():
                        img = Image.open(img_path)
                        st.image(img, use_column_width=True)
                        
        # Details & Debug Panel
        if "debug_trace" in result:
            with st.expander("🛠️ Show Step-by-Step AQU & Ingestion Debug Logs"):
                trace = result["debug_trace"]
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Query Classification")
                    st.json(trace["classification"])
                    
                    st.markdown("##### AQU Query Analysis")
                    st.json(trace["query_analysis"])
                    
                with col2:
                    st.markdown("##### Ranked Retrieval & Reranker Candidates")
                    st.json(trace["retrieved_results"])
                    
                    st.markdown("##### Answer Validation Details")
                    st.info(trace.get("validation_reason", "No details."))
