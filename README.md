# 👁️ ClearSight Analytics

> **Multimodal Clinical Decision Support System (CDSS) for Diabetic Care Management**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-red)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.1-f59e0b)](https://groq.com/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](LICENSE)

**ClearSight Analytics** is a high-fidelity multimodal AI platform designed to assist healthcare professionals in high-stakes clinical decision-making. It combines traditional machine learning, deep learning, computer vision, NLP, and generative AI to provide a 360-degree risk assessment of diabetic patients — from hospital readmission probability to retinal complication screening.

> ⚠️ **This is an academic Capstone Project.** It is not a certified medical device and must not be used for real clinical diagnosis.

---

## ✨ Key Features

- **5 Specialized AI Models** running in parallel on a single inference request
- **Grad-CAM Explainability** — visual heatmaps highlighting pathological retinal regions
- **AI Clinical Copilot** — context-aware LLM (Llama 3.1) that injects live patient metrics into every prompt
- **Quick Load Presets** — 5 pre-built clinical scenarios for rapid demo and testing
- **EHR Export** — one-click Markdown clinical summary ready for copy-paste into medical records
- **Clinical Safety Guardrails** — mandatory disclaimers, `temperature=0.2` lock, and OOD warnings on all AI outputs

---

## 🏛️ System Architecture

The platform operates on a **Consensus Inference Engine**, integrating five specialized models:

### Tier 1 — Predictive Risk (M1 & M2)
| Model | Type | Task |
|-------|------|------|
| **M1** | XGBoost Ensemble | 30-day readmission risk from structured EHR data |
| **M2** | Keras DNN | Parallel readmission probability for consensus scoring |

### Tier 2 — Computer Vision (M3)
| Model | Type | Task |
|-------|------|------|
| **M3** | ResNet50 CNN | Diabetic Retinopathy screening from fundus photographs |
| **Grad-CAM** | Explainable AI | Activation heatmaps over hemorrhages, exudates, and microaneurysms |

### Tier 3 — NLP & Operations (M4 & M5)
| Model | Type | Task |
|-------|------|------|
| **M4** | Meta LSTM (PyTorch) | Clinical note sentiment + qualitative risk signal detection |
| **M5** | Capacity Classifier | Length of Stay (LOS) prediction for bed management |

### Tier 4 — Generative AI Copilot
Powered by **Llama 3.1-8b-instant via Groq API** for sub-second latency. Dynamically injects live patient metrics and all five model outputs into the LLM context to generate personalized discharge protocols, risk driver rankings, and clinical syntheses.

---

## 🛡️ Clinical Safety & Ethics

| Guardrail | Implementation |
|-----------|---------------|
| **Hallucination control** | All generative outputs locked at `temperature: 0.2` |
| **Mandatory disclaimers** | Every AI response enforced with a `CRITICAL RULE` system instruction |
| **OOD detection UX** | Explicit warning on retinal uploader for non-fundus images |
| **No definitive diagnosis** | CDSS system prompt explicitly prohibits diagnostic conclusions |
| **Audit trail** | All prediction events logged via Python `logging` module |

---

## 📂 Project Structure

```
final-capstone-clearsight-analytics/
├── webapp/
│   └── app.py                    # Main Streamlit application (~5 000 lines)
├── models/
│   ├── model1_traditional_ml/    # XGBoost readmission ensemble
│   ├── model2_deep_learning/     # Keras DNN readmission model
│   ├── model3_cnn/               # ResNet50 retinal screening + Grad-CAM
│   ├── model4_nlp_classification/# Meta LSTM clinical note analysis
│   └── model5_innovation/        # LOS / capacity planning classifier
├── pipelines/                    # Data cleaning & feature engineering
├── notebooks/                    # EDA notebooks per team member
├── data/processed/               # Processed datasets (gitignored)
├── output_templates/             # Per-model result CSV templates
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Deployment

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com/) (for the AI Copilot)

### Setup

```bash
# 1. Clone
git clone https://github.com/fsa-aiml-2511/final-capstone-clearsight-analytics.git
cd final-capstone-clearsight-analytics

# 2. Create and activate virtual environment
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

### Secrets Configuration

Create `.streamlit/secrets.toml`:

```toml
GROQ_API_KEY = "gsk_your_key_here"
```

### Run

```bash
streamlit run webapp/app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 🛠️ Technical Stack

| Layer | Technology |
|-------|-----------|
| UI Framework | Streamlit 1.32+ with custom glassmorphism CSS |
| ML — Tabular | XGBoost, scikit-learn, joblib |
| ML — Deep Learning | TensorFlow / Keras 2.15 |
| ML — Computer Vision | PyTorch, torchvision (ResNet50), OpenCV |
| ML — NLP | PyTorch LSTM, custom tokenizer |
| Generative AI | Llama 3.1 via Groq API (OpenAI-compatible client) |
| Data Processing | pandas, NumPy, scikit-learn pipelines |
| Explainability | Grad-CAM (manual implementation, no third-party lib) |

---

## 👥 Team

| Role | Name | GitHub |
|------|------|--------|
| **ML / DNN Lead** | Francisco Molina | [@Frankmo89](https://github.com/Frankmo89) |
| **NLP Lead** | Wesley Houk | [@wesleyhouk](https://github.com/wesleyhouk) |
| **CNN / CV Lead** | Doug Bacon | [@Cooley5632] |
| **Data Engineering** | Francisco Molina | [@Frankmo89](https://github.com/Frankmo89) |

---

*⚠️ **Disclaimer:** This software is an academic Capstone Project for demonstration purposes only. It is not a certified medical device and should not be used for actual clinical diagnosis.*