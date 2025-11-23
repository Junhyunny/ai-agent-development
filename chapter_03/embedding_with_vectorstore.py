from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

texts = [
  "가장 중요한 건 눈에 보이지 않아. 마음으로 보아야만 비로소 잘 볼 수 있어. 어른들은 자꾸 숫자로만 세상을 보려고 해.",
  "파이썬(Python)은 간결하고 가독성이 높은 문법을 가진 프로그래밍 언어입니다. 'Life is short, You need Python'이라는 말이 있을 정도로 생산성이 높습니다.",
  "우리는 우주에 흔적을 남기기 위해 여기에 있습니다. 그렇지 않다면 도대체 왜 여기에 있겠습니까? 혁신은 리더와 추종자를 구분하는 잣대입니다.",
  "네가 나를 길들이면 우리는 서로에게 이 세상에서 오직 하나밖에 없는 존재가 될 거야. 네가 오후 4시에 온다면 난 3시부터 행복해지겠지.",
  "디버깅(Debugging)은 코드에서 버그를 찾아 없애는 과정입니다. 이는 마치 범죄 현장에서 단서를 찾는 탐정의 수사 과정과도 같습니다.",
  "지식보다 중요한 것은 상상력입니다. 지식은 한계가 있지만, 상상력은 세상의 모든 것을 끌어안을 수 있기 때문입니다.",
  "RAG(검색 증강 생성)는 LLM이 학습 데이터에 없는 최신 정보나 고유한 지식을 외부 데이터베이스에서 검색(Retrieval)하여 답변 생성에 활용하는 기술입니다.",
  "내 장미꽃이 그토록 소중한 이유는 내가 그 꽃을 위해 공들인 시간 때문이야. 너는 네가 길들인 것에 대해 영원히 책임을 져야 해.",
  "무엇이든 할 수 있거나, 할 수 있다고 꿈꾸는 것이 있다면 시작하라. 대담함 속에는 천재성, 힘, 그리고 마법이 숨어 있다.",
  "벡터 데이터베이스(Vector Database)는 텍스트나 이미지를 고차원의 숫자 배열(임베딩)로 변환하여 저장합니다. 이를 통해 키워드가 달라도 의미가 유사한 데이터를 찾아낼 수 있습니다.",
]

vectorstore = FAISS.from_texts(texts, embeddings)

query = "힘이 나는 명언 알려주세요."
docs = vectorstore.similarity_search(query, k=3)

print("검색 결과: ")
for i, doc in enumerate(docs):
  print(f"{i + 1}. {doc.page_content}")
