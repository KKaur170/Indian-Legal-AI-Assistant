import streamlit as st
import PyPDF2 
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------------------------------------------------------
# 1. Custom AI Architecture for IPC Prediction
# ---------------------------------------------------------
class LegalExpertModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return {"logits": logits}

class IPCClassifier:
    def __init__(self, model_dir="./ipc_section"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the Answer Key (The Laws)
        self.classes = np.load(f"{model_dir}/ipc_classes.npy", allow_pickle=True)
        num_labels = len(self.classes)
        
        # Load the Optimized Math Thresholds
        self.thresholds = np.load(f"{model_dir}/optimal_thresholds.npy")
        
        # Load Tokenizer & Model Weights
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = LegalExpertModel("nlpaueb/legal-bert-base-uncased", num_labels)
        self.model.load_state_dict(torch.load(f"{model_dir}/custom_model_weights.bin", map_location=self.device))
        self.model.to(self.device)
        self.model.eval() 

    def predict(self, text_scenario):
        encodings = self.tokenizer(text_scenario, truncation=True, padding='max_length', max_length=512, return_tensors="pt")
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"].squeeze()
            
        probabilities = torch.sigmoid(logits).cpu().numpy()
        predicted_laws = []
        if probabilities.ndim == 0:
            probabilities = np.array([probabilities])
            
        for i, prob in enumerate(probabilities):
            if prob > self.thresholds[i]:
                predicted_laws.append({
                    "ipc_section": self.classes[i],
                    "confidence": round(float(prob * 100), 2)
                })
                
        if not predicted_laws:
            return [{"ipc_section": "No crime detected or insufficient details", "confidence": 0.0}]
        return sorted(predicted_laws, key=lambda x: x['confidence'], reverse=True)


# ---------------------------------------------------------
# 2. Setup Page & CSS
# ---------------------------------------------------------
st.set_page_config(page_title="Indian Legal Assistant", page_icon="⚖️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] { background-color: #0b1325; font-family: 'Times New Roman', Times, serif; }
    h1, h2, h3 { color: #d4af37 !important; } 
    p, li, label, .stMarkdown { color: #e2e8f0; font-family: sans-serif; }
    .stDeployButton {display:none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] { background-color: #151e32 !important; border-right: 1px solid #d4af37; padding-top: 10px; }
    .sb-card { background-color: #1c2742; border: 1px solid #374151; border-radius: 8px; padding: 15px; margin-top: 15px; color: #9ca3af; transition: transform 0.2s, border-color 0.2s; }
    .sb-card-header { color: #d4af37; font-weight: bold; font-size: 1em; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }
    .stChatMessage { border-radius: 12px; margin-bottom: 15px; padding: 10px;}
    .stChatMessage:nth-child(even) { background-color: #151e32; border: 1px solid #374151; }
    .stChatMessage:nth-child(odd) { background-color: #0b1325; border: 1px dashed #d4af37; }
    [data-testid="stChatInput"] { border-radius: 20px !important; background-color: #151e32 !important; border: 1px solid #d4af37 !important; }
    [data-testid="stChatInput"] input { color: #e2e8f0 !important; }
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# 3. Model Loaders (Global Memory)
# ---------------------------------------------------------
@st.cache_resource
def load_text_classifier():
    try: return pipeline("text-classification", model="./legal_brain")
    except Exception: return None

@st.cache_resource
def load_ipc_classifier():
    try: return IPCClassifier(model_dir="./ipc_section")
    except Exception: return None

@st.cache_resource
def load_summarizer():
    try: 
        # FINAL FIX: Direct Engine Loading (No more pipeline crashes!)
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        return {"tokenizer": tokenizer, "model": model}
    except Exception as e: 
        print(f"Summarizer Error: {e}")
        return None

legal_classifier = load_text_classifier()
ipc_engine = load_ipc_classifier()
summarizer_engine = load_summarizer()


# ---------------------------------------------------------
# 4. Sidebar (File Upload & API)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<br><h2 style='text-align: center; margin-top: 15px;'>⚖️ Workspace</h2>", unsafe_allow_html=True)
    st.markdown("""<div class='sb-card'><div class='sb-card-header'>🔑 API Configuration</div></div>""", unsafe_allow_html=True)
    api_key = st.text_input("Enter Google Gemini API Key", type="password", placeholder="AIzaSy...")

    uploaded_file = st.file_uploader("Upload Legal Document (PDF)", type="pdf") 
    
    if uploaded_file is not None:
        file_name = uploaded_file.name
        st.success(f"Loaded: {file_name}")
        
        if "current_file" not in st.session_state or st.session_state.current_file != file_name:
            with st.spinner("Extracting & Vectorizing..."):
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                extracted_text = "".join([page.extract_text() + "\n" for page in pdf_reader.pages if page.extract_text()])
                st.session_state.document_text = extracted_text
                st.session_state.current_file = file_name
            
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                text_chunks = text_splitter.split_text(extracted_text)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vector_store = FAISS.from_texts(text_chunks, embeddings)
                st.session_state.messages = [{"role": "assistant", "content": f"New document '{file_name}' loaded! Ask me anything."}]
    else:
        st.info("Awaiting PDF upload...")
        for key in ["document_text", "current_file", "vector_store"]:
            if key in st.session_state: del st.session_state[key]
    
    # --- TEAM CREDITS ---
    st.markdown("<br><hr style='border-top: 1px solid #d4af37;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #d4af37; text-align: center;'>Project Engineers</h4>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size: 0.9em; color: #e2e8f0; text-align: center;'>
        <b>Deep Learning & AI Architecture:</b><br> Khushnoor Kaur<br><br>
        <b>UI & System Integration:</b><br> Sandeep Kaur
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# 5. Main UI & Tabs
# ---------------------------------------------------------
st.markdown("<h1>Indian Legal Document Assistant</h1><br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["💬 Document Q&A", "🔍 Sentence Classifier", "⚖️ Predict IPC Section", "📄 AI Summarizer"])

# --- TAB 1: RAG Chat ---
with tab1:
    st.markdown("### Ask questions based on the uploaded document")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome! Please upload a document and enter your API key."}]

    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "⚖️"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question..."):
        with st.chat_message("user", avatar="🧑‍💻"): st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        if not api_key:
            st.warning("Please enter your Gemini API Key in the sidebar.")
        elif "vector_store" not in st.session_state:
            st.warning("Please upload a PDF document first.")
        else:
            with st.chat_message("assistant", avatar="⚖️"):
                with st.spinner("Analyzing..."):
                    docs = st.session_state.vector_store.similarity_search(prompt, k=4)
                    context_text = "\n\n".join([doc.page_content for doc in docs])
                    legal_prompt = f"""You are a legal AI. Answer strictly based on the context. If not found, say so. Do not hallucinate.\n\nContext:\n{context_text}\n\nQuestion:\n{prompt}"""
                    try:
                        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
                        response = llm.invoke(legal_prompt)
                        st.markdown(response.content)
                        st.session_state.messages.append({"role": "assistant", "content": response.content})
                    except Exception as e:
                        st.error(f"Error: {e}")

# --- TAB 2: Text Classifier ---
with tab2:
    st.markdown("### Identify Legal Category (Contextual)")
    
    # 1. Create two separate input boxes to respect the model's training architecture
    prev_sentence = st.text_area("Context / Previous Sentence (Highly recommended):", height=68, key="tab2_prev")
    curr_sentence = st.text_area("Target Sentence to Classify:", height=68, key="tab2_curr")
    
    if st.button("Classify Contextual Text"):
        if curr_sentence and legal_classifier:
            with st.spinner("Analyzing context..."):
                try:
                    # 2. Combine them exactly how the AI was trained!
                    if prev_sentence.strip():
                        combined_text = f"{prev_sentence.strip()} [SEP] {curr_sentence.strip()}"
                    else:
                        # Fallback just in case they leave context blank
                        combined_text = curr_sentence.strip()
                        
                    # 3. Feed the properly structured string to the pipeline
                    res = legal_classifier(combined_text)[0]
                    raw_label = res['label']
                    
                    label_map = {"LABEL_0": "Analysis", "LABEL_1": "Argument (Petitioner)", "LABEL_2": "Argument (Respondent)", "LABEL_3": "Fact (FAC)", "LABEL_4": "Issue", "LABEL_5": "None", "LABEL_6": "Preamble", "LABEL_7": "Precedent (Not Relied)", "LABEL_8": "Precedent (Relied)", "LABEL_9": "Ratio Decidendi", "LABEL_10": "Ruling by Lower Court", "LABEL_11": "Ruling by Present Court", "LABEL_12": "Statute (Law)"}
                    
                    final_label = label_map.get(raw_label, raw_label)
                    st.success(f"**{final_label}** ({round(res['score'] * 100, 2)}%)")
                except Exception as e: 
                    st.error(f"Error: {e}")
        elif not curr_sentence: 
            st.warning("Please enter a target sentence to classify.")
        else:
            st.error("Model failed to load. Please check your model files.")

# --- TAB 3: IPC Predictor ---
with tab3:
    st.markdown("### Predict Relevant IPC Sections")
    ipc_input = st.text_area("Enter Crime Scenario:", height=150)
    if st.button("Predict IPC Charges", type="primary"):
        if ipc_input and ipc_engine:
            with st.spinner("Consulting AI Expert Engine..."):
                try:
                    predictions = ipc_engine.predict(ipc_input)
                    st.markdown("#### 🚨 Predicted Charges:")
                    for res in predictions:
                        law = res['ipc_section']
                        conf = res['confidence']
                        if conf == 0.0: st.info("No relevant IPC section detected.")
                        else: st.error(f"**{law}** | Confidence: {conf}%")
                except Exception as e: st.error(f"Error: {e}")
        else: st.warning("Please describe a scenario.")

# --- TAB 4: Document Summarizer ---
with tab4:
    st.markdown("### 📄 Local AI Text Summarizer")
    st.markdown("<p style='font-size:0.9em; color:#cbd5e1;'>Paste long legal paragraphs here to get a concise summary.</p>", unsafe_allow_html=True)
    
    summ_input = st.text_area("Legal Text to Summarize:", height=200, placeholder="Paste lengthy legal arguments or facts here...")
    
    if st.button("Generate Summary", type="primary", key="summ_btn"):
        if summ_input and summarizer_engine:
            with st.spinner("Generating Summary... (This may take 10-15 seconds)"):
                try:
                    tokenizer = summarizer_engine["tokenizer"]
                    model = summarizer_engine["model"]
                    
                    # Convert Text to Tokens
                    inputs = tokenizer(summ_input, return_tensors="pt", max_length=1024, truncation=True)
                    
                    # Generate Summary
                    summary_ids = model.generate(
                        inputs["input_ids"], 
                        max_length=150, 
                        min_length=40, 
                        length_penalty=1.0, 
                        num_beams=4, 
                        early_stopping=True
                    )
                    
                    # Convert Tokens back to English Text
                    final_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    
                    st.markdown("#### ✨ Final Summary:")
                    st.success(final_summary)
                except Exception as e:
                    st.error(f"Summarization failed. Error: {e}")
        elif not summarizer_engine:
            st.error("Summarizer model not loaded properly. Check the 'legal_summarizer' folder.")
        else:
            st.warning("Please paste some text first.")