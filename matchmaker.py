import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from huggingface_hub import hf_hub_download
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Streamlit page config
st.set_page_config(page_title="Drug Repurposing Matchmaker", layout="wide")

# ---------------------------
# 0) Config via Streamlit secrets / env
# ---------------------------
HF_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN")
HF_REPO_ID = st.secrets.get("HF_REPO_ID", "TheBloke/Llama-2-7B-Chat-GGUF")
HF_FILENAME = st.secrets.get("HF_FILENAME", "llama-2-7b-chat.Q4_K_M.gguf")

# Optional tuning
LLM_CTX = int(st.secrets.get("LLM_CTX", 4096))
LLM_TEMPERATURE = float(st.secrets.get("LLM_TEMPERATURE", 0.7))
LLM_MAX_TOKENS = int(st.secrets.get("LLM_MAX_TOKENS", 256))
LLM_N_GPU_LAYERS = int(st.secrets.get("LLM_N_GPU_LAYERS", 0))  # >0 only if your wheel supports Metal/CUDA
LLM_N_THREADS = int(st.secrets.get("LLM_N_THREADS", 0))        # 0 = auto

# ---------------------------
# 1) Download (or reuse cached) GGUF from Hugging Face
# ---------------------------
@st.cache_resource(show_spinner="Downloading model from Hugging Face...")
def get_model_path() -> str:
    """
    Downloads the GGUF model file (resumes + caches).
    Returns the local path to the file.
    """
    if HF_TOKEN is None and ("meta-llama" in HF_REPO_ID or "Llama-2" in HF_REPO_ID):
        raise RuntimeError(
            "HUGGINGFACE_TOKEN is required for gated repos. "
            "Add it to Streamlit secrets."
        )

    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        token=HF_TOKEN,              # uses token for gated repos; None is fine for open repos
        local_files_only=False,
        resume_download=True
    )
    return path

# ---------------------------
# 2) Load the llama.cpp model once
# ---------------------------
@st.cache_resource(show_spinner="Loading LLM…")
def load_llm(model_path: str) -> ChatLlamaCpp:
    return ChatLlamaCpp(
        model_path=model_path,
        n_ctx=LLM_CTX,
        n_gpu_layers=LLM_N_GPU_LAYERS,
        n_threads=LLM_N_THREADS,
        temperature=LLM_TEMPERATURE,
        verbose=False,
        # You can also set stop=["</s>", "User:", "Assistant:"] if needed
        max_tokens=LLM_MAX_TOKENS,  # default cap; you can change per session via secrets
    )

# ---------------------------
# 3) Build the LangChain pipeline
# ---------------------------
SYSTEM_MSG = "You are a helpful, concise assistant."
prompt_tmpl = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", "{input}")
])

@st.cache_resource(show_spinner=False)
def get_chain(_llm: ChatLlamaCpp):
    # simple chain: prompt -> model -> string
    return prompt_tmpl | _llm | StrOutputParser()

# ---------------------------
# 4) Query helper (drop-in function)
# ---------------------------
def query_ollama(prompt_text: str, max_new_tokens: int = None) -> str:
    """
    Torch-free, no-Ollama query using LangChain + llama.cpp.
    If max_new_tokens is given, we recreate a lightweight chain with that limit.
    """
    try:
        model_path = get_model_path()
        llm = load_llm(model_path)

        if max_new_tokens is not None and max_new_tokens != LLM_MAX_TOKENS:
            # For per-call token limits, create a shallow new llm wrapper
            llm_dynamic = ChatLlamaCpp(
                model_path=model_path,
                n_ctx=LLM_CTX,
                n_gpu_layers=LLM_N_GPU_LAYERS,
                n_threads=LLM_N_THREADS,
                temperature=LLM_TEMPERATURE,
                verbose=False,
                max_tokens=max_new_tokens,
            )
            chain = prompt_tmpl | llm_dynamic | StrOutputParser()
            return chain.invoke({"input": prompt_text}).strip()

        # Default: use cached chain
        chain = get_chain(llm)
        return chain.invoke({"input": prompt_text}).strip()

    except Exception as e:
        return f"(LLM unavailable) Mock narrative: {str(e)}"


# ---------------------------
# 2. Load Kaggle Drug Repositioning CSVs
# ---------------------------
@st.cache_data
def load_data():
    diseases = pd.read_csv("diseasesInfo.csv")  # DiseaseID, DiseaseName, etc.
    drugs = pd.read_csv("drugsInfo.csv")        # DrugID, DrugName, DrugTarget, etc.
    mapping = pd.read_csv("mapping.csv")        # DrugID, DiseaseID
    return diseases, drugs, mapping

diseases_df, drugs_df, mapping_df = load_data()

# ---------------------------
# 3. Build drug-target mapping
# ---------------------------
# Merge DrugTarget info from drugsInfo
mapping_df = mapping_df.merge(drugs_df[["DrugID", "DrugTarget", "DrugName"]], on="DrugID", how="left")

# drug_id -> set of targets
drug_to_targets = mapping_df.groupby("DrugID")["DrugTarget"].apply(lambda x: set([t for t in x if pd.notna(t)])).to_dict()
drug_id_to_name = drugs_df.set_index("DrugID")["DrugName"].to_dict()
disease_id_to_name = diseases_df.set_index("DiseaseID")["DiseaseName"].to_dict()

# Universe of targets
all_targets = sorted({t for targets in drug_to_targets.values() for t in targets})
drug_list = list(drug_to_targets.keys())

# Binary drug x target matrix
drug_gene_df = pd.DataFrame(0, index=drug_list, columns=all_targets, dtype=int)
for d, targets in drug_to_targets.items():
    for t in targets:
        if t in drug_gene_df.columns:
            drug_gene_df.at[d, t] = 1
drug_vectors = drug_gene_df.values

# ---------------------------
# 4. Similarity / scoring functions
# ---------------------------
def vector_for_targets(targets):
    return np.array([1 if t in targets else 0 for t in all_targets], dtype=int)

def score_similarity(vec1, vec2):
    sim = cosine_similarity(vec1.reshape(1,-1), vec2.reshape(1,-1))[0,0]
    return (sim + 1)/2  # normalize 0-1

def top_matches(disease_targets, top_k=5):
    disease_vec = vector_for_targets(disease_targets)
    scores = [(d, score_similarity(drug_vectors[i], disease_vec)) for i, d in enumerate(drug_list)]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# ---------------------------
# 5. Streamlit Dashboard
# ---------------------------

st.title("💊 AI Molecule Matchmaker")

st.markdown("""
This tool finds potential repurposing candidates for a disease from existing drugs using their targets.
**Now powered by LangChain + Ollama for intelligent explanations.**
""")

# Disease selection
disease_name = st.selectbox("Select a disease:", sorted(diseases_df["DiseaseName"].tolist()))
disease_id = diseases_df[diseases_df["DiseaseName"] == disease_name]["DiseaseID"].values[0]

# Targets for this disease
disease_targets = mapping_df[mapping_df["DiseaseID"] == disease_id]["DrugTarget"].dropna().unique().tolist()

if len(disease_targets) == 0:
    st.warning("No known targets for this disease. Showing top random drugs instead.")
    results_df = drugs_df.sample(5)
    results_df["score"] = np.random.rand(len(results_df))
else:
    # Compute top matches
    matches = top_matches(disease_targets, top_k=5)
    results_df = pd.DataFrame({
        "drug_name": [drug_id_to_name[d[0]] for d in matches],
        "score": [d[1] for d in matches]
    })

# Show top matches
st.subheader(f"Top candidate drugs for **{disease_name}**")
st.table(results_df)

# Simple Plotly scatter for demo
st.subheader("Drug Compatibility Map (mock layout)")
np.random.seed(42)
results_df["x"] = np.random.rand(len(results_df))
results_df["y"] = np.random.rand(len(results_df))
fig = px.scatter(results_df, x="x", y="y", text=results_df.get("drug_name", results_df.get("DrugName","")),
                 color="score", size="score", size_max=20, title="Top Drug Matches")
st.plotly_chart(fig, use_container_width=True)

# Generate LLM narratives with caching
@st.cache_data(show_spinner="Generating AI explanations...")
def get_narrative(drug, disease):
    prompt = f"Explain in simple terms why {drug} could be repurposed for {disease}."
    return query_ollama(prompt)

st.subheader("AI-Powered Drug Repurposing Explanations")
st.info("💡 Using LangChain with local Ollama for intelligent biomedical explanations")

for idx, row in results_df.iterrows():
    drug_name = row["drug_name"]
    score = row["score"]
    
    with st.expander(f"🧬 {drug_name} (Match Score: {score:.2f})"):
        with st.spinner(f"Generating explanation for {drug_name}..."):
            narrative = get_narrative(drug_name, disease_name)
            st.write(narrative)
            
            # Add some visual separation
            st.markdown("---")
            st.caption(f"*AI explanation generated for {drug_name} → {disease_name}*")

# Sidebar with Ollama status
with st.sidebar:
    st.header("🔧 Configuration")
    st.info("""
    **LangChain + Ollama Setup:**
    - Using local Ollama instance
    - LangChain for prompt management
    - Better error handling
    - Improved response quality
    """)
    
    # Model selection
    model_option = st.selectbox(
        "Ollama Model:",
        ["llama2", "llama2:13b", "medllama2", "codellama"],
        help="Select which Ollama model to use for explanations"
    )
    
    if st.button("🔄 Test Ollama Connection"):
        with st.spinner("Testing connection..."):
            test_response = query_ollama("Say 'Hello' in one word.", model=model_option)
            if "unavailable" in test_response.lower():
                st.error("❌ Ollama not connected")
                st.code("Run: ollama serve", language="bash")
            else:
                st.success("✅ Ollama connected successfully")
                st.code(f"Response: {test_response}")

st.caption("Data from Kaggle Drug Repositioning dataset. Narratives generated by LangChain + Ollama.")