import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
# 1. Read document
with open("data.txt", "r", encoding="utf-8") as file:
    text = file.read()

# 2. Split document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
chunks = splitter.split_text(text)
print("Total Chunks:", len(chunks))

# 3. Create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

# 4. Store chunks in FAISS
vector_store = FAISS.from_texts(
    chunks,
    embedding=embeddings
)
print("\nVectors stored successfully!")

# 5. Ask a question
question = "What is RAG?"

# 6. Similarity search
results = vector_store.similarity_search(
    question,
    k=2
)

# 7. Display retrieved chunks
print("\nQuestion:")
print(question)
print("\nRetrieved Information:")
for i, result in enumerate(results):
    print(f"\nResult {i+1}:")
    print(result.page_content)
