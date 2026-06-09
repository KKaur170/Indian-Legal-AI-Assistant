<div align="center">
  
#  Indian Legal AI Assistant (ILAA)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-FF4B4B?logo=streamlit&logoColor=white)

An advanced, multi-pipeline deep learning architecture designed to automate the extraction of legal intelligence from highly complex, unstructured Indian Supreme Court judgments and legal documents.

**[ Live Hugging Face Deployment](https://huggingface.co/spaces/KKaur17/Indian-Legal-AI-Assistant)** | **[ GitHub Repository](https://github.com/KKaur170/Indian-Legal-AI-Assistant)**

</div>

---

##  Executive Summary

The Indian judicial system is characterized by a massive volume of pending cases and deeply complex, unstructured legal documents. Legal professionals spend countless hours manually reading through lengthy case laws, FIRs, and judgments to extract core facts, determine applicable statutes, and understand judicial reasoning.

This project delivers an AI-powered Legal Assistant tailored specifically for the structural and linguistic nuances of Indian legal text. Rather than relying entirely on generic out-of-the-box LLMs, the platform orchestrates multiple specialized Transformer pipelines—each engineered and optimized for a specific legal intelligence task via custom loss functions, dynamic thresholding, and sequence-aware pretraining.

---

##  Platform Interface

*(Note: The system is deployed via a Streamlit-based interactive control center capable of orchestrating multiple AI pipelines simultaneously. Images are stored in the `assets/` directory.)*

| Document Q&A (RAG Pipeline) | Contextual Sentence Classifier |
| :---: | :---: |
| ![Home](assets/home.png) | ![Classifier](assets/classifier.png) |
| **Gemini 2.5 Flash + FAISS Vector Search** | **InLegalBERT + Focal Loss** |

| Multi-Label IPC Charge Predictor | AI Legal Summarizer |
| :---: | :---: |
| ![IPC Predictor](assets/ipc.png) | ![Summarizer](assets/summarizer.png) |
| **Legal-BERT + Dynamic Threshold Matrix** | **Zero-Shot Abstractive Summarization** |

---

##  Core AI Pipelines & Engineering Highlights

### 1. Context-Aware Rhetorical Role Classification (Tab 2)
Analyzes individual sentences within Indian Supreme Court judgments and classifies them into 13 distinct rhetorical roles (e.g., *RATIO, PRECEDENT, STATUTE, ANALYSIS*).
* **Foundational Model:** `law-ai/InLegalBERT`
* **Engineering Highlight (Sequence Pair Classification):** Standard classifiers process sentences independently, which fails in legal text where meaning relies heavily on historical context. We engineered a sequence pair architecture (`[Previous Sentence] [SEP] [Current Sentence]`) to provide sequential narrative memory to the model.
* **Loss Function Optimization:** Addressed severe long-tail class imbalance by replacing traditional Cross-Entropy with **Multi-Class Focal Loss (γ = 2.0)**, aggressively penalizing misclassified minority examples like `STATUTE`.

### 2. Multi-Label IPC Prediction (Document-Level) (Tab 3)
Predicts multiple overlapping Indian Penal Code (IPC) sections directly from unstructured crime scenarios.
* **Foundational Model:** `nlpaueb/legal-bert-base-uncased`
* **Engineering Highlight (Dynamic Threshold Calibration):** Bypassed the standard `0.5` binary probability cutoff. Engineered a custom optimization framework that evaluated validation logits to generate an `optimal_thresholds.npy` matrix, selecting the optimal trigger threshold independently for 101 distinct IPC classes. This mathematical optimization alone yielded a **+10.54% Absolute F1 Gain**.

### 3. Retrieval-Augmented Generation (RAG) (Tab 1)
Allows users to upload extensive legal PDFs and query them conversationally without hallucination.
* **Extraction & Chunking:** `PyPDF2` combined with LangChain's `RecursiveCharacterTextSplitter`.
* **Vector Store:** Semantic embeddings stored in a local `FAISS` database powered by `all-MiniLM-L6-v2`.
* **Generation:** Context is retrieved and injected into Gemini 2.5 Flash for precise, document-grounded responses.

### 4. Zero-Shot Legal Summarization (Tab 4)
Generates concise legal summaries while maintaining document privacy through local inference using `facebook/bart-large-cnn`. Implemented beam search constraints (`num_beams = 4`, `early_stopping = True`) to maintain narrative coherence and judicial chronology.

---

##  Repository Structure

```text
Indian-Legal-AI-Assiatant/
├── assets/                    # UI screenshots and system diagrams
│   ├── home.png
│   ├── classifier.png
│   ├── ipc.png
│   └── summarizer.png
├── notebooks/                 # Core Machine Learning Training Pipelines
│   ├── IPC_Section_Training.ipynb
│   └── Text_Classification_Training.ipynb
├── src/
│   ├── app.py                 # Streamlit frontend UI and session state logic
│   └── inference.py           # Custom PyTorch model loading and prediction
├── requirements.txt           # Environment dependencies
├── .gitignore                 # Excludes heavy .bin/.safetensors weights
└── README.md                  # Project documentation

```
*(Note: Custom .bin and .safetensors model weights exceed GitHub storage limits and are securely hosted within the Hugging Face deployment environment.)*

---

##  Local Installation & Setup
### 1. Clone the Repository:

```bash
git clone https://github.com/KKaur170
cd Indian-Legal-AI-Assistant
```

### 2. Create a Virtual Environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies:

```bash
pip install -r requirements.txt
```

### 4. Launch the Application:

```bash
streamlit run src/app.py
```

---

##  Technology Stack
* **Deep Learning Frameworks:** PyTorch, Hugging Face Transformers, Datasets
* **NLP & Vector Operations:** LangChain, FAISS, Sentence-Transformers
* **Large Language Models:** Gemini 2.5 Flash, BART
* **Frontend UI:** Streamlit
* **Hardware utilized for training:** NVIDIA T4 Tensor Core GPU (Mixed Precision FP16)

