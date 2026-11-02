import os

import streamlit as st

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (

    GoogleGenerativeAIEmbeddings,

    ChatGoogleGenerativeAI

)

from langchain_community.vectorstores import FAISS

 

# Load API key

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

 

# Streamlit title

st.title(" Simple RAG Application")

st.write("Ask questions from our document")

 

# Read document

with open("data.txt", "r", encoding="utf-8") as file:

    text = file.read()

 

# Split document

splitter = RecursiveCharacterTextSplitter(

    chunk_size=100,

    chunk_overlap=20

)

chunks = splitter.split_text(text)

 

# Create embeddings

embeddings = GoogleGenerativeAIEmbeddings(

    model="models/gemini-embedding-001",

    google_api_key=api_key

)

 

# Create FAISS vector store

vector_store = FAISS.from_texts(

    chunks,

    embedding=embeddings

)

 

# User input

question = st.text_input("Ask your question:")

 

if st.button("Get Answer"):

    if question:

        # Retrieve relevant chunks

        results = vector_store.similarity_search(

            question,

            k=2

        )

        # Combine chunks

        context = "\n".join(

            [result.page_content for result in results]

        )

        # Create prompt

        prompt = f"""

        Answer the question using only the context below.

        Context:

        {context}

        Question:

        {question}

        Answer:

        """

        # Gemini

        llm = ChatGoogleGenerativeAI(

            model="gemini-2.5-flash",

            google_api_key=api_key

        )

        # Generate answer

        response = llm.invoke(prompt)

        # Display answer

        st.subheader("Answer")

        st.write(response.content)

    else:

        st.warning("Please enter a question.")