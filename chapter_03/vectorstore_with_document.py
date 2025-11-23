from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()

embeddings = OpenAIEmbeddings()
text_spilitter = CharacterTextSplitter(separator=".", chunk_size=50, chunk_overlap=20)

documents = [
  Document(
    page_content="파이썬(Python)은 간결하고 가독성이 높은 문법을 가진 프로그래밍 언어입니다. 'Life is short, You need Python'이라는 격언이 유명합니다.",
    metadata={"source": "dev_guide_v1.txt", "topic": "programming"},
  ),
  Document(
    page_content="가장 중요한 건 눈에 보이지 않아. 마음으로 보아야만 비로소 잘 볼 수 있어. 어른들은 자꾸 숫자로만 세상을 판단하려고 해.",
    metadata={"source": "little_prince.txt", "topic": "literature"},
  ),
  Document(
    page_content="RAG(검색 증강 생성)는 LLM이 학습하지 않은 외부 데이터를 검색하여 답변의 정확도를 높이는 기술입니다. 환각 현상을 줄이는 데 효과적입니다.",
    metadata={"source": "ai_trends_2024.pdf", "topic": "technology"},
  ),
  Document(
    page_content="네가 나를 길들이면 우리는 서로에게 이 세상에서 오직 하나밖에 없는 존재가 될 거야. 네가 오후 4시에 온다면 난 3시부터 행복해지겠지.",
    metadata={"source": "little_prince.txt", "topic": "literature"},
  ),
  Document(
    page_content="우리가 우주에 흔적을 남기기 위해 여기에 있는 것입니다. 혁신은 리더와 추종자를 구분하는 잣대입니다.",
    metadata={"source": "steve_jobs_bio.txt", "topic": "inspiration"},
  ),
]

split_docs = text_spilitter.split_documents(documents)
for doc in split_docs:
  print(
    f"문서: {doc.page_content[:50]}... | 출처: {doc.metadata['source']} | 주제: {doc.metadata['topic']}"
  )

vectorstore = FAISS.from_documents(documents, embeddings)

query = "초보자가 배우기 좋은 프로그래밍 언어?"
results = vectorstore.similarity_search(query, k=2)

print(f"질문: {query}")
print("검색 결과: ")
for i, doc in enumerate(results, 1):
  print(f"{i}. {doc.page_content[:100]}...")
  print(f"  출처: {doc.metadata['source']}")
  print(f"  주제: {doc.metadata['topic']}")


results = vectorstore.similarity_search_with_score(query, k=2)
print("\n\n유사도 점수: ")
for doc, score in results:
  print(f"{doc.metadata['source']} - {score:.3f}")
