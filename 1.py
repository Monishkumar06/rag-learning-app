from langchain_text_splitters import RecursiveCharacterTextSplitter

# Read our document

with open("data.txt", "r", encoding="utf-8") as file:

    text = file.read()

print("Original Document:")

print(text)

# Create text splitter

splitter = RecursiveCharacterTextSplitter(

    chunk_size=100,

    chunk_overlap=20

)

# Split document

chunks = splitter.split_text(text)

print("\nTotal Chunks:", len(chunks))

# Print each chunk

for i, chunk in enumerate(chunks):

    print(f"\nChunk {i+1}:")

    print(chunk)