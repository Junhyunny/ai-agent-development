import json
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()


class MemoryBotState(BaseModel):
  user_message: str = Field(default="", description="사용자 입력 메시지")
  user_name: str = Field(default="", description="사용자 이름")
  user_preferences: Dict[str, Any] = Field(
    default_factory=dict, description="사용자 선호도"
  )
  response: str = Field(default="", description="최중 응답")


llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")


def process_message(state: MemoryBotState) -> Dict[str, Any]:
  message = state.user_message
  user_name = state.user_name
  preferences = state.user_preferences.copy()

  system_prompt = f"""
  당신은 사용자의 정보를 기억하는 메모리 봇입니다.
  현재 기억하는 정보:
  - 사용자 이름: {user_name if user_name else "모름"}
  - 좋아하는 것: {preferences.get("likes", [])}
  - 실어하는 것: {preferences.get("dislikes", [])}
  사용자 메시지를 분석하여 다음 JSON 형식으로 응답하세요. 코드 블럭 표시는 필요 없습니다.:
  {{
    "response": "사용자에게 줄 응답 메시지",
    "new_name": "새로 알게 된 이름(없으면 null)",
    "new_likes": ["새로 알게 된 좋아하는 것들"],
    "new_dislikes": ["새로 알게 된 싫어하는 것들"]
  }}
  """

  messages = [SystemMessage(content=system_prompt), HumanMessage(content=message)]
  response = llm.invoke(input=messages)
  result = json.loads(response.content)

  if result.get("new_name"):
    user_name = result["new_name"]

  if result.get("new_likes"):
    preferences.setdefault("likes", []).extend(result["new_likes"])

  if result.get("new_dislikes"):
    preferences.setdefault("dislikes", []).extend(result["new_dislikes"])

  bot_response = result.get("response", "죄송해요, 이해하지 못했어요.")

  return {
    "response": bot_response,
    "user_name": user_name,
    "user_preferences": preferences,
  }


def create_memory_bot_graph():
  checkpointer = InMemorySaver()
  workflow = StateGraph(MemoryBotState)

  workflow.add_node("process_message_node", process_message)

  workflow.add_edge(START, "process_message_node")
  workflow.add_edge("process_message_node", END)

  return workflow.compile(checkpointer=checkpointer)


def main():
  print("=== InMemorySaver 메모리 봇 테스트 ===\n")

  app = create_memory_bot_graph()
  thread_id = "jun-123"

  conversations = [
    "안녕하세요.",
    "내 이름은 준현이야.",
    "김치찜을 좋아해",
    "해삼은 싫어해",
    "내 이름이 뭐였지?",
    "내가 좋아하는 것과 싫어하는 것은?",
  ]

  for i, message in enumerate(conversations, 1):
    print(f"[{i}] 사용자: {message}")
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"user_message": message}, config)
    print(f"[{i}] 챗봇: {result['response']}")
    print(f"이름={result.get('user_name', '없음')}")
    print(f"좋아하는 것={result.get('user_preferences', {})}\n")

  image = app.get_graph().draw_mermaid_png()
  with open("./03_persistent_memory.png", "wb") as f:
    f.write(image)


if __name__ == "__main__":
  main()
