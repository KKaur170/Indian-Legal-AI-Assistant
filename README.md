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

**The Problem:** The Indian judicial system processes millions of incredibly dense, unstructured documents. Legal professionals spend countless billable hours manually reading case laws, FIRs, and judgments just to extract basic facts, find relevant laws, and understand the judge's reasoning.

**The Solution:** We built an AI platform that acts as a junior legal associate. Instead of relying on generic chatbots (which often hallucinate laws), this platform uses specialized, custom-trained Machine Learning models engineered specifically for the language of the Indian Supreme Court. It can instantly summarize 100-page rulings, predict criminal charges from plain-English scenarios, and classify judicial text with high precision.

---

##  Platform Interface & Features

*(Note: The system is deployed via a Streamlit-based interactive control center capable of orchestrating multiple AI pipelines simultaneously.)*

### 1. Document Q&A (RAG Pipeline)
**Functionality:** Upload a massive legal PDF and chat with it. Ask "What was the final ruling?" and get an instant, fact-checked answer.

**Technical Implementation:** Powered by `PyPDF2`, LangChain, FAISS Vector Search, and Gemini 2.5 Flash.
<img src="assets/home.png" width="800" alt="Home Dashboard">

<br>

### 2. Contextual Sentence Classifier
**Functionality:** Reads a legal document like a human does—remembering the context of the previous sentence to accurately determine if the judge is stating a Fact, citing a Precedent, or making a Ruling.

**Technical Implementation:** Implements **Sequence Pair Classification** (`[Previous] [SEP] [Current]`) on `law-ai/InLegalBERT`. We replaced standard Cross-Entropy with **Multi-Class Focal Loss (γ = 2.0)** to aggressively correct class imbalances.
<img src="assets/classifier.png" width="800" alt="Sentence Classifier">

<br>

### 3. Multi-Label IPC Charge Predictor
**Functionality:** Type in a plain-English description of a crime, and the AI instantly predicts all applicable Indian Penal Code (IPC) sections with confidence scores.

**Technical Implementation:** Fine-tuned `nlpaueb/legal-bert`. Engineered a custom optimization framework that bypasses the standard `0.5` binary cutoff, utilizing an `optimal_thresholds.npy` matrix to trigger 101 distinct classes independently. **This mathematical optimization alone yielded a +10.54% Absolute F1 Gain.**
<img src="assets/ipc.png" width="800" alt="IPC Predictor">

<br>

### 4. AI Legal Summarizer
**Functionality:** Condenses lengthy, confusing legal arguments into short, easy-to-read summaries in seconds.

**Technical Implementation:** Zero-shot abstractive summarization via `facebook/bart-large-cnn`, utilizing constrained beam search (`num_beams = 4`) to maintain judicial chronology without hallucination.
<img src="assets/summarizer.png" width="800" alt="Summarizer">

---

##  Core AI Pipelines & Engineering Highlights

### 1. Retrieval-Augmented Generation (RAG) (Tab 1)
Allows users to upload extensive legal PDFs and query them conversationally without hallucination.

* **Extraction & Chunking:** `PyPDF2` combined with LangChain's `RecursiveCharacterTextSplitter`.
* **Vector Store:** Semantic embeddings stored in a local `FAISS` database powered by `all-MiniLM-L6-v2`.
* **Generation:** Context is retrieved and injected into Gemini 2.5 Flash for precise, document-grounded responses.

### 2. Context-Aware Rhetorical Role Classification (Tab 2)
Analyzes individual sentences within Indian Supreme Court judgments and classifies them into 13 distinct rhetorical roles (e.g., *RATIO, PRECEDENT, STATUTE, ANALYSIS*).

* **Foundational Model:** `law-ai/InLegalBERT`
* **Engineering Highlight (Sequence Pair Classification):** Standard classifiers process sentences independently, which fails in legal text where meaning relies heavily on historical context. We engineered a sequence pair architecture (`[Previous Sentence] [SEP] [Current Sentence]`) to provide sequential narrative memory to the model.
* **Loss Function Optimization:** Addressed severe long-tail class imbalance by replacing traditional Cross-Entropy with **Multi-Class Focal Loss (γ = 2.0)**, aggressively penalizing misclassified minority examples like `STATUTE`.

### 3. Multi-Label IPC Prediction (Document-Level) (Tab 3)
Predicts multiple overlapping Indian Penal Code (IPC) sections directly from unstructured crime scenarios.

* **Foundational Model:** `nlpaueb/legal-bert-base-uncased`
* **Engineering Highlight (Dynamic Threshold Calibration):** Bypassed the standard `0.5` binary probability cutoff. Engineered a custom optimization framework that evaluated validation logits to generate an `optimal_thresholds.npy` matrix, selecting the optimal trigger threshold independently for 101 distinct IPC classes. This mathematical optimization alone yielded a **+10.54% Absolute F1 Gain**.

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

---

##  Future Improvements

* **Full-Document Context Windowing:** Upgrading from sequence-pairs to global document architectures to process entire multi-page legal files at once.
* **100% IPC Coverage:** Scaling the multi-label prediction pipeline to encompass *every* section of the Indian Penal Code, moving beyond the current top 100.
* **Model Expansion:** Benchmarking newer, long-context LLMs and specialized Transformers to boost legal extraction accuracy.
* **Relational Crime Mapping:** Building analytical features to visually map the dependencies and co-occurrences between different criminal offenses.

---

##  Author & AI Architect

**Khushnoor Kaur** 
*B.E. Computer Engineering | Thapar Institute of Engineering and Technology*

Spearheaded the Deep Learning architecture, model training pipelines, and core Natural Language Processing math for this platform. Primary engineering contributions include:
* Designing the Sequence-Pair Contextual Window for Legal-BERT.
* Engineering the mathematical optimization of the Multi-Class Focal Loss and dynamic threshold matrices.
* Leading the end-to-end model training, evaluation, and architectural strategy for the predictive NLP subsystems.

**Let's Connect:** [GitHub](https://github.com/KKaur170) | [LinkedIn](https://www.linkedin.com/in/khushnoor-kaur-bb7684345/)

---

##  License & Acknowledgements
This project is licensed under the MIT License. Developed for academic research, experimentation, and educational purposes. Special thanks to the open-source communities and research teams behind Hugging Face, PyTorch, InLegalBERT, and Law-AI for enabling continued innovation in Legal AI.
