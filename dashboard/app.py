import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.data_loader import load_fitbit_data
from src.preprocessing.data_cleaning import clean_health_data
from src.preprocessing.feature_engineering import create_features
from src.faiss_index.faiss_manager import load_vector_store, search_vector_store
from src.models.anomaly_detector import detect_anomalies
from src.models.time_series_model import LSTMModel, predict_future
from src.llm.assistant import HealthAssistant

st.set_page_config(
    page_title="VitalMind AI Health Dashboard",
    page_icon="🩺",
    layout="wide"
)

@st.cache_data
def run_preprocessing_pipeline(df):
    cleaned_df = clean_health_data(df)
    final_df = create_features(cleaned_df)
    final_df['Date'] = pd.to_datetime(final_df['Date'])
    return final_df

@st.cache_resource
def load_prediction_model(model_path):
    model = LSTMModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_faiss_db(index_folder_path):
    return load_vector_store(index_folder_path)

@st.cache_resource
def load_assistant(data_path, index_folder_path):
    return HealthAssistant(data_path, index_folder_path)

st.title("🩺 VitalMind AI: Personal Health Insight Platform")


base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_DATA_PATH = os.path.join(base_path, 'data', 'sample_fitbit_data.csv')
PROCESSED_DATA_PATH = os.path.join(base_path, 'data', 'processed_health_data.csv')
MODEL_PATH = os.path.join(base_path, 'models', 'lstm_heart_rate.pth')
INDEX_FOLDER_PATH = os.path.join(base_path, 'models', 'faiss_index')


uploaded_file = st.sidebar.file_uploader("Upload your health data (CSV)", type="csv")
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
else:
    raw_df = load_fitbit_data(DEFAULT_DATA_PATH)

if raw_df is not None:
    final_df = run_preprocessing_pipeline(raw_df)
    prediction_model = load_prediction_model(MODEL_PATH)
    faiss_db = load_faiss_db(INDEX_FOLDER_PATH)

    st.header("Your Health Overview")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "⚠️ Anomaly Detection", "🔍 Similarity Search", "🔮 Forecasting", "💬 AI Assistant"])
    
    with tab1:
        st.subheader("Key Health Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Steps", f"{final_df['Steps'].mean():.0f}")
        col2.metric("Avg Resting Heart Rate", f"{final_df['RestingHeartRate'].mean():.0f} bpm")
        col3.metric("Avg Sleep Hours", f"{final_df['SleepHours'].mean():.1f} hrs")
        
        st.subheader("Metrics Over Time")
        st.line_chart(final_df.set_index('Date')[['Steps', 'RestingHeartRate', 'SleepHours']])

    with tab2:
        st.subheader("Anomaly Detection")
        df_anomalies = detect_anomalies(final_df)
        anomalous_days = df_anomalies[df_anomalies['is_anomaly']]
        if not anomalous_days.empty:
            st.warning(f"Found {len(anomalous_days)} potential anomalous day(s):")
            st.dataframe(anomalous_days[['Date', 'Steps', 'RestingHeartRate', 'SleepHours', 'DayOfWeek']])
        else:
            st.success("No anomalies detected in this period.")

    with tab3:
        st.subheader("Find Similar Health Days")
        if faiss_db:
            selected_day_index = st.selectbox(
                "Select a day to find similar days:",
                options=final_df.index,
                format_func=lambda i: final_df.loc[i, 'Date'].strftime('%Y-%m-%d')
            )
            
            if st.button("Find Similar Days"):
                row = final_df.loc[selected_day_index]
                query_text = (
                    f"On {row['DayOfWeek']}, activity level was {row['ActivityLevel']} with {row['Steps']} steps. "
                    f"Resting heart rate was {row['RestingHeartRate']:.0f} bpm. "
                    f"Sleep was {row['SleepQuality']} with {row['SleepHours']:.1f} hours."
                )

                results = search_vector_store(faiss_db, query_text, k=4)

                similar_sentences = [doc.page_content for doc in results]

                similar_days_df = final_df[final_df.apply(lambda r: (
                    f"On {r['DayOfWeek']}, activity level was {r['ActivityLevel']} with {r['Steps']} steps. "
                    f"Resting heart rate was {r['RestingHeartRate']:.0f} bpm. "
                    f"Sleep was {r['SleepQuality']} with {r['SleepHours']:.1f} hours."
                ) in similar_sentences, axis=1)]

                st.write(f"Days most similar to **{final_df.loc[selected_day_index, 'Date'].strftime('%Y-%m-%d')}**:")
                display_df = similar_days_df[similar_days_df.index != selected_day_index]
                st.dataframe(display_df[['Date', 'Steps', 'RestingHeartRate', 'SleepHours', 'DayOfWeek']].head(3))
        else:
            st.error("FAISS vector store not found. Please run the embeddings_generator.py script.")

    with tab4:
        st.subheader("Resting Heart Rate Forecast")
        data = final_df['RestingHeartRate'].values.astype(float)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_normalized = scaler.fit_transform(data.reshape(-1, 1))
        future_steps = 7
        predictions_normalized = predict_future(prediction_model, data_normalized, steps=future_steps)
        predictions = scaler.inverse_transform(np.array(predictions_normalized).reshape(-1, 1))
        last_date = final_df['Date'].max()
        future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, future_steps + 1)])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=final_df['Date'], y=final_df['RestingHeartRate'], mode='lines', name='Historical Data'))
        fig.add_trace(go.Scatter(x=future_dates, y=predictions.flatten(), mode='lines', name='Forecast', line=dict(dash='dash')))
        fig.update_layout(title='Resting Heart Rate: 7-Day Forecast', xaxis_title='Date', yaxis_title='Heart Rate (bpm)')
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Chat with your VitalMind AI Assistant")
        assistant = load_assistant(PROCESSED_DATA_PATH, INDEX_FOLDER_PATH)
        
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

else:
    st.error("Could not load data. Please check the default data path or upload a file.")
