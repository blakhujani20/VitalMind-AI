import os
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings 

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HealthAssistant:
    def __init__(self, data_path, index_folder_path, model_name='all-MiniLM-L6-v2'):
        try:
            self.df = pd.read_csv(data_path)
        except FileNotFoundError:
            raise

        self.embeddings_model = SentenceTransformerEmbeddings(model_name=model_name)

        try:
            self.vector_store = FAISS.load_local(
                folder_path=index_folder_path,
                embeddings=self.embeddings_model,
                allow_dangerous_deserialization=True 
            )
        except Exception as e:
            raise e

        self.retriever = self.vector_store.as_retriever()
        self.rag_chain = self._create_rag_chain()

    def _create_rag_chain(self):
        template = """
        You are a helpful and insightful personal health assistant.
        Use the following pieces of retrieved health data context to answer the user's question.
        If you don't know the answer, just say that you don't know.
        Provide a concise, helpful summary based on the data.

        Context: {context}
        Question: {question}

        Helpful Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def answer_question(self, question):
        """Answers a user's question using the RAG chain."""
        return self.rag_chain.invoke(question)


if __name__ == '__main__':
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed_health_data.csv')
    FAISS_INDEX_FOLDER = os.path.join(PROJECT_ROOT, 'models', 'faiss_index')

    try:
        assistant = HealthAssistant(data_path=PROCESSED_DATA_PATH, index_folder_path=FAISS_INDEX_FOLDER)

        question1 = "Summarize my health trends for the last few days."
        answer1 = assistant.answer_question(question1)

        question2 = "How was my sleep on the day I had the most steps?"
        answer2 = assistant.answer_question(question2)

    except Exception as e:
        print(f"\nAn error occurred during assistant initialization: {e}")

