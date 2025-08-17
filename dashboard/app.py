import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import pickle
import plotly.graph_objects as go
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.data_loader import load_raw_data
from src.preprocessing.data_cleaning import clean_and_merge_data
from src.preprocessing.feature_engineering import create_features
from src.faiss_index.faiss_manager import load_vector_store, search_vector_store
from src.models.anomaly_detector import detect_anomalies
from src.models.time_series_model import LSTMModel, predict_future
from src.llm.assistant import HealthAssistant
from src.faiss_index.embeddings_generator import generate_and_save_embeddings

st.set_page_config(page_title="VitalMind AI", page_icon="🩺", layout="wide")

@st.cache_data
def run_preprocessing_pipeline(activity_df, sleep_df):
    cleaned_df = clean_and_merge_data(activity_df, sleep_df)
    final_df = create_features(cleaned_df)
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    return final_df

@st.cache_resource
def get_vector_store(_df):
    vector_store = generate_and_save_embeddings(_df, save_path=None)
    return vector_store

@st.cache_resource
def load_prediction_model(model_path):
    model = LSTMModel(hidden_layer_size=50)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)

def display_dashboard(df):
    st.header("Your Health Overview")
    
    prediction_model = load_prediction_model(MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)
    vector_store = get_vector_store(df)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "⚠️ Anomaly Detection", "🔍 Similarity Search", "🔮 Forecasting", "💬 AI Assistant"])

    with tab1:
        st.subheader("Key Health Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Steps", f"{df['Steps'].mean():.0f}")
        col2.metric("Avg Calories", f"{df['Calories'].mean():.0f}")
        col3.metric("Avg Sleep (hrs)", f"{df['SleepHours'].mean():.1f}")
        
        st.subheader("Metrics Over Time (Sampled for one user if multiple exist)")
        sample_user_id = df['Id'].unique()[0]
        sample_df = df[df['Id'] == sample_user_id]
        st.line_chart(sample_df.set_index('Date')[['Steps', 'Calories', 'SleepHours']])

    with tab2:
        st.subheader("Anomaly Detection")
        df_anomalies = detect_anomalies(df)
        anomalous_days = df_anomalies[df_anomalies['is_anomaly']]
        if not anomalous_days.empty:
            st.warning(f"Found {len(anomalous_days)} potential anomalous day(s):")
            st.dataframe(anomalous_days[['Date', 'Steps', 'Calories', 'SleepHours', 'DayOfWeek']])
        else:
            st.success("No anomalies detected in this period.")

    with tab3:
        st.subheader("Find Similar Health Days")
        if vector_store:
            query_text = st.text_input("Describe a day to find similar ones (e.g., 'high steps and good sleep')", "a day with high steps and optimal sleep")
            if st.button("Find Similar Days"):
                results = search_vector_store(vector_store, query_text, k=3)
                st.write(f"Days most similar to: '{query_text}'")
                for doc in results:
                    st.info(doc.page_content)

    with tab4:
        st.subheader("Daily Steps Forecast")
        data = df['Steps'].values.astype(float)
        if len(data) > 14:
            data_normalized = scaler.transform(data.reshape(-1, 1))
            future_steps = 7
            look_back = 14 
            predictions_normalized = predict_future(prediction_model, data_normalized, steps=future_steps, look_back=look_back)
            predictions = scaler.inverse_transform(np.array(predictions_normalized).reshape(-1, 1))
            
            last_date = df['Date'].max()
            future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, future_steps + 1)])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Steps'], mode='lines', name='Historical Data'))
            fig.add_trace(go.Scatter(x=future_dates, y=predictions.flatten(), mode='lines', name='Forecast', line=dict(dash='dash')))
            fig.update_layout(title='Daily Steps: 7-Day Forecast', xaxis_title='Date', yaxis_title='Number of Steps')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data to generate a forecast. Please provide at least 15 days of records.")

    with tab5:
        st.subheader("Chat with your VitalMind AI Assistant")
        assistant = HealthAssistant(vector_store, df)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        if prompt := st.chat_input("Ask about your health trends..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Thinking..."):
                response = assistant.answer_question(prompt)
                
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

st.title("🩺 VitalMind AI: Personal Health Insight Platform")

if 'app_mode' not in st.session_state:
    st.session_state.app_mode = 'welcome'
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(base_path, 'data')
MODEL_PATH = os.path.join(base_path, 'models', 'lstm_steps_model.pth')
SCALER_PATH = os.path.join(base_path, 'models', 'scaler_steps.pkl')


if st.session_state.app_mode == 'welcome':
    st.header("Welcome to VitalMind AI")
    st.markdown("Your personal health data, transformed into actionable intelligence. Choose an option to begin:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔬 Explore a Demo Dataset", use_container_width=True):
            activity_df, sleep_df = load_raw_data(DATA_PATH)
            if activity_df is not None:
                st.session_state.processed_df = run_preprocessing_pipeline(activity_df, sleep_df)
                st.session_state.app_mode = 'dashboard'
                st.rerun()
            else:
                st.error("Default data files not found. Please check the /data directory.")
    
    with col2:
        if st.button("⬆️ Analyze My Own Data", use_container_width=True):
            st.session_state.app_mode = 'user_upload'
            st.rerun()

elif st.session_state.app_mode == 'user_upload':
    st.sidebar.header("Upload Your Data")
    activity_file = st.sidebar.file_uploader("Upload Daily Activity CSV", type="csv")
    sleep_file = st.sidebar.file_uploader("Upload Daily Sleep CSV", type="csv")
    
    if activity_file and sleep_file:
        try:
            activity_df = pd.read_csv(activity_file)
            sleep_df = pd.read_csv(sleep_file)
            st.session_state.processed_df = run_preprocessing_pipeline(activity_df, sleep_df)
            st.session_state.app_mode = 'dashboard'
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error processing files: {e}")
    
    if st.button("← Go Back"):
        st.session_state.app_mode = 'welcome'
        st.rerun()

elif st.session_state.app_mode == 'dashboard':
    if st.session_state.processed_df is not None:
        display_dashboard(st.session_state.processed_df)
        if st.sidebar.button("Start Over"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
    else:
        st.error("No data loaded. Please return to the welcome screen.")
        if st.button("← Go Back"):
            st.session_state.app_mode = 'welcome'
            st.rerun()