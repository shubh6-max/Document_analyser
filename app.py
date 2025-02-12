import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import PyPDF2
import io

# -------------------------------
# Configuration and Setup
# -------------------------------
st.set_page_config(page_title="LLM Document Insight Extractor", layout="wide")

# Custom CSS for Modern Aesthetics
st.markdown(
    """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f4f4f4;
    }
    .main-title {
        font-size: 2.8em;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.2em;
    }
    .subtitle {
        font-size: 1.3em;
        color: #34495e;
        margin-bottom: 1.5em;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-size: 1em;
        padding: 0.6em 1.2em;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        color: white;
    }
    .response-box {
        background-color: #ecf0f1;
        border-radius: 8px;
        padding: 1.2em;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
        margin-top: 1em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Load Model and Tokenizer
# -------------------------------
model_path = "meta-llama/Llama-3.2-3B-Instruct"

@st.cache_resource
def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, device_map="cpu")
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer(model_path)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("📄 Document Insight Extractor")
st.sidebar.markdown(
    """**Instructions:**
    1. Upload a document (PDF or text-based file).
    2. Review the extracted text.
    3. Provide a prompt for analysis (e.g., "Summarize key insights.").
    4. Click "Generate Response" to get AI-driven insights.
    """
)
st.sidebar.write("---")
st.sidebar.markdown("**About:** This app leverages a local LLM to analyze documents and extract insights.")

# -------------------------------
# Main Page Layout
# -------------------------------
st.markdown('<div class="main-title">LLM Document Insight Extractor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload documents and let AI generate actionable insights</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------------
# File Uploader and Processing
# -------------------------------
st.write("### Step 1: Upload Your Document")
uploaded_file = st.file_uploader("", type=["pdf", "txt"])

if uploaded_file is not None:
    # Process PDF or text file
    all_text = ""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            all_text += page.extract_text() + "\n"
    elif uploaded_file.type == "text/plain":
        all_text = uploaded_file.read().decode("utf-8")
    
    # Show Extracted Text
    st.write("### Step 2: Review Extracted Text (Optional)")
    with st.expander("View Extracted Text"):
        st.text_area("", all_text, height=200)
    
    st.write("### Step 3: Provide a Prompt for the LLM")
    user_prompt = st.text_input("Ask the LLM something about the document content:", 
                                value="Summarize the key insights of the document in bullet points.")
    generate_button = st.button("Generate Response")

    if generate_button:
        prompt = f"The following is the content of a document:\n\n{all_text}\n\n{user_prompt}"
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with st.spinner("Generating response..."):
            outputs = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=True, 
                temperature=0.7, 
                top_p=0.9, 
                top_k=50
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the initial document content from the response
            response = response.split("The following is the content of a PDF:")[-1].strip()
        
        st.write("### Step 4: LLM Response")
        st.markdown(f"<div class='response-box'>{response}</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='instructions'>No document uploaded yet. Please upload a document to proceed.</div>", unsafe_allow_html=True)
