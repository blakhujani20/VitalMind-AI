<div align="center">

# VitalMind AI 🩺🤖

**Transforming wearable health data into actionable wellness insights with AI.**

</div>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-008638?style=for-the-badge">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
</p>

---

## 🎯 About The Project

Wearables like Fitbit generate vast amounts of data — but without interpretation, it’s just numbers.  
**VitalMind AI** bridges this gap by turning raw activity and sleep data into **meaningful health narratives**, **predictive insights**, and an **AI companion** for personal wellness.

This project demonstrates a **full-stack AI application pipeline**, from raw data preprocessing and model training, to deployment with an interactive dashboard and conversational AI.

---

## ✨ Key Features

- **📊 In-Depth Health Analysis**  
  Clean and merge wearable datasets, perform **EDA** to reveal trends and correlations in steps, sleep, and calories.

- **🔮 Predictive Forecasting**  
  Train a **PyTorch LSTM time-series model** to forecast daily steps and evaluate performance against baselines.

- **⚠️ Anomaly Detection**  
  Detect outlier days using an **Isolation Forest**, flagging potential unusual activity or health conditions.

- **🔍 Similarity Search**  
  Encode daily summaries into vector embeddings with **Sentence Transformers**, store in a **FAISS index**, and query similar health days instantly.

- **💬 Conversational AI Assistant**  
  A **RAG pipeline** built with **LangChain** + a local **Ollama Llama-3** model that answers natural language health questions.

- **📊 Interactive Dashboard**  
  **Streamlit app** for visualizing insights, forecasting trends, and chatting with the AI assistant.

---

## 🚀 Demo & Screenshots

📌 **Live Demo:** (Coming Soon)  


## 🛠️ Tech Stack

| Category         | Tools / Frameworks                                                                 |
| ---------------- | ---------------------------------------------------------------------------------- |
| **Data Science** | `Pandas`, `NumPy`, `Scikit-learn`, `Matplotlib`, `Seaborn`                         |
| **AI / ML**      | `PyTorch`, `Sentence Transformers`, `FAISS`, `Isolation Forest`                    |
| **GenAI**        | `LangChain`, `Ollama` (Llama-3)                                                    |
| **App / UI**     | `Streamlit`, `Plotly`                                                              |
| **Dev Tools**    | `Git`, `GitHub`, `VS Code`, `Conda`, `Jupyter`                                     |
| **Deployment**   | `AWS (S3, Lambda, API Gateway)` or `Streamlit Cloud`                               |

---

## ⚙️ Getting Started

Follow these steps to set up and run VitalMind AI locally.

### 1️⃣ Prerequisites
- Python 3.9+
- Conda (or venv/Poetry)
- [Ollama](https://ollama.com/) installed and running
- Kaggle dataset: [Fitbit Fitness Tracker Data](https://www.kaggle.com/datasets/arashnic/fitbit)

### 2️⃣ Setup Instructions
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/VitalMind-AI.git
cd VitalMind-AI

# Create environment
conda create --name vitalmind python=3.9
conda activate vitalmind

# Install dependencies
pip install -r requirements.txt

# Pull local LLM for RAG assistant
ollama pull llama3:8b
```

### 3️⃣ Dataset
- Download from Kaggle → Fitbit dataset.
- Place dailyActivity_merged.csv and sleepDay_merged.csv inside the /data folder.

### 4️⃣ Run Pipelines
- Train LSTM predictive model ```python training_model.py```

- Build FAISS index for similarity search  ```python embedding_pipeline.py```
  

### 5️⃣ Run App
- ```streamlit run dashboard/app.py```
  
---

### 📜 License
- Distributed under the MIT License.