import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

words = ["강아지", "고양이", "자동차", "비행기"]
word_embeddings = embeddings.embed_documents(words)

query = "동물"
query_embedding = embeddings.embed_query(query)


def cosine_similarity(a, b):
  dot_product = np.dot(a, b)
  norm_a = np.linalg.norm(a)
  norm_b = np.linalg.norm(b)
  return dot_product / (norm_a * norm_b + 1e-9)


print(f"'{query}'에 대한 유사도: ")
for word, embedding in zip(words, word_embeddings):
  similarity = cosine_similarity(query_embedding, embedding)
  print(f"{word}: {similarity:.3f}")
