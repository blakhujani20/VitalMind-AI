import os
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM as Ollama
from langchain_huggingface import HuggingFaceEmbeddings as SentenceTransformerEmbeddings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class HealthAssistant:
    def __init__(self, vector_store, df):
        if df is None or vector_store is None:
            raise ValueError("DataFrame and vector_store must be provided.")
        
        self.df = df
        self.vector_store = vector_store
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
        llm = Ollama(model="phi3:mini")

        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def answer_question(self, question):
        return self.rag_chain.invoke(question)
