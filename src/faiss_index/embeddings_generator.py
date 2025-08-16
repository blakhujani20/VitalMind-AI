from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')


def generate_embeddings(df):
    if df is None:
        return None, None

    sentences = []
    for index, row in df.iterrows():
        sentence = (
            f"On {row['DayOfWeek']}, activity level was {row['ActivityLevel']} with {row['Steps']} steps. "
            f"Resting heart rate was {row['RestingHeartRate']:.0f} bpm. "
            f"Sleep was {row['SleepQuality']} with {row['SleepHours']:.1f} hours."
        )
        sentences.append(sentence)

    embeddings = model.encode(sentences, show_progress_bar=True)
    return sentences, embeddings