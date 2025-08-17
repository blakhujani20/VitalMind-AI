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

st.set_page_config(page_title="VitalMind AI Health Dashboard", page_icon="🩺", layout="wide")
@st.cache_data
def run_preprocessing_pipeline(activity_df, sleep_df):
    cleaned_df = clean_and_merge_data(activity_df, sleep_df)
    final_df = create_features(cleaned_df)
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    return final_df
@st.cache_resource
def load_prediction_model(model_path):
    model = LSTMModel(hidden_layer_size=50)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model
@st.cache_resource
def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler
@st.cache_resource
def load_faiss_db(index_folder_path):
    return load_vector_store(index_folder_path)
@st.cache_resource
def load_assistant(data_path, index_folder_path):
    return HealthAssistant(data_path, index_folder_path)

st.title("🩺 VitalMind AI: Personal Health Insight Platform")
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(base_path, 'data')
PROCESSED_DATA_PATH = os.path.join(base_path, 'data', 'fitbit_data_processed.csv')
MODEL_PATH = os.path.join(base_path, 'models', 'lstm_steps_model.pth')
SCALER_PATH = os.path.join(base_path, 'models', 'scaler_steps.pkl')
INDEX_FOLDER_PATH = os.path.join(base_path, 'models', 'faiss_index')

st.sidebar.header("Upload Your Data")
st.sidebar.info("Upload both daily activity and daily sleep CSV files from the fitbit app.")
activity_file = st.sidebar.file_uploader("Upload your Daily Activity CSV", type="csv")
sleep_file = st.sidebar.file_uploader("Upload your Daily Sleep CSV", type="csv")

use_default_data = True
if activity_file is not None and sleep_file is not None:
    try:
        activity_df = pd.read_csv(activity_file)
        sleep_df = pd.read_csv(sleep_file)
        st.sidebar.success("Files uploaded successfully! Displaying your data.")
        use_default_data = False
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded files: {e}")
        activity_df, sleep_df = load_raw_data(DATA_PATH) # Fallback
else:
    st.sidebar.info("No files uploaded. Displaying default Kaggle dataset.")
    activity_df, sleep_df = load_raw_data(DATA_PATH)


if activity_df is not None and sleep_df is not None:
    final_df = run_preprocessing_pipeline(activity_df, sleep_df)
    prediction_model = load_prediction_model(MODEL_PATH)
    scaler = load_scaler(SCALER_PATH)
    faiss_db = load_faiss_db(INDEX_FOLDER_PATH)

    st.header("Your Health Overview")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "⚠️ Anomaly Detection", "🔍 Similarity Search", "🔮 Forecasting", "💬 AI Assistant"])
    
    with tab1:
        st.subheader("Key Health Metrics (Aggregated from all Users)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Steps", f"{final_df['Steps'].mean():.0f}")
        col2.metric("Avg Calories Burned", f"{final_df['Calories'].mean():.0f}")
        col3.metric("Avg Sleep Hours", f"{final_df['SleepHours'].mean():.1f} hrs")
        
        st.subheader("Metrics Over Time (Sample User)")
        sample_user_id = final_df['Id'].unique()[0]
        sample_df = final_df[final_df['Id'] == sample_user_id]
        st.line_chart(sample_df.set_index('Date')[['Steps', 'Calories', 'SleepHours']])

    with tab2:
        st.subheader("Anomaly Detection")
        df_anomalies = detect_anomalies(sample_df)
        anomalous_days = df_anomalies[df_anomalies['is_anomaly']]
        if not anomalous_days.empty:
            st.warning(f"Found {len(anomalous_days)} potential anomalous day(s) for sample user:")
            st.dataframe(anomalous_days[['Date', 'Steps', 'Calories', 'SleepHours', 'DayOfWeek']])
        else:
            st.success("No anomalies detected for sample user in this period.")

    with tab3:
        st.subheader("Find Similar Health Days (for Sample User)")
        if faiss_db:
            selected_day_index = st.selectbox(
                "Select a day to find similar days:",
                options=sample_df.index,
                format_func=lambda i: sample_df.loc[i, 'Date'].strftime('%Y-%m-%d')
            )
            
            if st.button("Find Similar Days"):
                row = sample_df.loc[selected_day_index]
                query_text = (
                    f"On {row['DayOfWeek']}, the activity level was {row['ActivityLevel']} with {row['Steps']:.0f} steps and {row['Calories']:.0f} calories burned. "
                    f"Sleep was {row['SleepQuality']} with {row['SleepHours']:.1f} hours."
                )

                results = search_vector_store(faiss_db, query_text, k=4)
                st.write(f"Days most similar to **{row['Date'].strftime('%Y-%m-%d')}**:")
                
                similar_contents = [doc.page_content for doc in results]
                st.write(similar_contents)

    with tab4:
        st.subheader("Daily Steps Forecast (for Sample User)")
        
        data = sample_df['Steps'].values.astype(float)
        data_normalized = scaler.transform(data.reshape(-1, 1))
  
        future_steps = 7
        look_back = 14 
        predictions_normalized = predict_future(prediction_model, data_normalized, steps=future_steps, look_back=look_back)
        predictions = scaler.inverse_transform(np.array(predictions_normalized).reshape(-1, 1))
        
        last_date = sample_df['Date'].max()
        future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, future_steps + 1)])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample_df['Date'], y=sample_df['Steps'], mode='lines', name='Historical Data'))
        fig.add_trace(go.Scatter(x=future_dates, y=predictions.flatten(), mode='lines', name='Forecast', line=dict(dash='dash')))
        fig.update_layout(title='Daily Steps: 7-Day Forecast', xaxis_title='Date', yaxis_title='Number of Steps')
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Chat with your VitalMind AI Assistant")
        assistant = load_assistant(PROCESSED_DATA_PATH, INDEX_FOLDER_PATH)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        if prompt := st.chat_input("Ask about health trends in the dataset..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Thinking..."):
                response = assistant.answer_question(prompt)
                
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.error("Could not load data. Please check the default data path in the /data folder.")