<div align="center">

# VitalMind AI 🤖

**Transforming wearable health data into actionable wellness insights with a deployed AI application.**

[![Live App](https://img.shields.io/badge/Live_App-Visit_Now-brightgreen?style=for-the-badge)](http://13.232.254.221:8501/)

</div>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-blue?style=for-the-badge&logo=python">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-008638?style=for-the-badge">
  <img alt="AWS" src="https://img.shields.io/badge/AWS-orange?style=for-the-badge&logo=amazonaws">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
</p>

---

## 🎯 About The Project

Wearables like Fitbit generate vast amounts of data—but without interpretation, it’s just numbers.
**VitalMind AI** bridges this gap by turning raw activity and sleep data into **meaningful health narratives**, **predictive insights**, and an **AI companion** for personal wellness.

This project demonstrates a **full-stack AI application pipeline**, from raw data preprocessing and model training to cloud deployment with an interactive dashboard and a high-speed conversational AI.

---
## ✨ Key Features

- **📊 In-Depth Health Analysis**: Clean and merge wearable datasets, performing **EDA** to reveal trends and correlations in steps, sleep, and calories.
- **🔮 Predictive Forecasting**: Train a **PyTorch LSTM time-series model** to forecast daily steps and evaluate performance against baselines.
- **⚠️ Anomaly Detection**: Detect outlier days using an **Isolation Forest**, flagging potential unusual activity or health conditions.
- **🔍 Similarity Search**: Encode daily summaries into vector embeddings with **Sentence Transformers**, store in a **FAISS index**, and query similar health days instantly.
- **💬 High-Speed Conversational AI**: A **RAG pipeline** built with **LangChain** and powered by the **Groq API** (Llama3) to answer natural language health questions with near-instant responses.
- **☁️ Cloud-Deployed Dashboard**: An interactive **Streamlit app** for visualizing insights and chatting with the AI, deployed on **AWS** and accessible to anyone.

---

## 🚀 Live Demo

You can interact with the live application here:
**[http://13.232.254.221:8501/](http://13.232.254.221:8501/)**


---

## 🛠️ Tech Stack

| Category         | Tools & Frameworks                                      |
| ---------------- | ------------------------------------------------------- |
| **Data Science** | `Pandas`, `NumPy`, `Scikit-learn`                       |
| **AI / ML** | `PyTorch`, `Sentence Transformers`, `FAISS`             |
| **GenAI** | `LangChain`, `Groq` (Llama3)                        |
| **App / UI** | `Streamlit`, `Plotly`                                   |
| **Deployment** | `AWS`, `tmux`, `Git`                                |

---

## ☁️ Deployment Architecture

The application is deployed on a **t3.micro** instance on **AWS EC2**, running an Ubuntu Server OS.

-   **Networking**: An AWS Security Group is configured to allow inbound traffic on port `22` (for SSH access) and `8501` (for the Streamlit app).
-   **Environment**: The PyTorch installation is CPU-specific to optimize for the `t3.micro` hardware and stay within the small 8GB disk storage limit.
-   **Process Management**: The **`tmux`** terminal multiplexer is used to run the Streamlit application in a persistent background session, ensuring the app stays live even after the SSH connection is closed.

---

## ⚙️ Getting Started (Local Setup)

Follow these steps to set up and run VitalMind AI on your own machine.

### 1️⃣ Prerequisites
-   Python 3.10+
-   A free **Groq API Key**. Get one [here](https://groq.com/).
-   Kaggle dataset: [Fitbit Fitness Tracker Data](https://www.kaggle.com/datasets/arashnic/fitbit)

### 2️⃣ Setup Instructions

```bash
# 1. Clone the repository
git clone [https://github.com/blakhujani20/VitalMind-AI.git](https://github.com/blakhujani20/VitalMind-AI.git)
cd VitalMind-AI

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate
# On Windows, use: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your API key
# Create a file named .env in the root directory
# and add your Groq API key to it like this:
# GROQ_API_KEY="gsk_YourSecretKeyHere"
```

### 3️⃣ Dataset
- Download from Kaggle → Fitbit dataset.
- Place dailyActivity_merged.csv and sleepDay_merged.csv inside the /data folder.

### 4️⃣ Run Pipelines
```bash
# Train the LSTM predictive model
python training_model.py

# Build the FAISS index for similarity search
python embedding_pipeline.py
```

### 5️⃣ Run App
```bash
streamlit run dashboard/app.py
```  
---

### 📜 License
- Distributed under the MIT License.