import faiss
import numpy as np

#Our Stored vectors
vectors = np.array([
[1.0,1.0], # Apple
[1.2,1.1], # Mango
[8.0,9.0], #Car
],dtype="float32")

# create Faiss Index
index = faiss.IndexFlatL2(2)

# Add Vectors
index .add(vectors)

#Search vectors
query = np.array([[1.1,1.0]],dtype="float32")

#find 2 nearest vectors
distances, indices = index.search(query, k=2)
print(indices)
print(distances)



https://github.com/Monishkumar06/rag-learning-app