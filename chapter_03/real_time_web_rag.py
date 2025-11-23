import time

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


class RealTimeWebRag:
  def __init__(self):
    self.search = DuckDuckGoSearchResults()
    self.llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    message = """
      웹에서 검색한 최신 정보를 바탕으로 답변하세요.
      검색 결과:
      {search_results}
      질문:
      {question}
      중요: 검색 결과에 있는 정보만 사용해서 답변하세요.
    """
    self.qa_prompt = ChatPromptTemplate.from_messages([("human", message)])

  def answer(self, question: str):
    """실시간 검색 후 답변"""
    print(f"검색 중: {question}")
    search_results = self.search.run(question)
    time.sleep(5)  # rate limit 에러 방지를 위한 대기
    qa_chain = self.qa_prompt | self.llm
    return qa_chain.invoke({"question": question, "search_results": search_results})


wab_rag = RealTimeWebRag()

questions = ["오늘 주요 뉴스는?", "오늘 야구 순위는?", "최신 AI 기술 동향은?"]

for q in questions:
  print(f"\n질문: {q}")
  answer = wab_rag.answer(q)
  print(f"답변: {answer.content}")
