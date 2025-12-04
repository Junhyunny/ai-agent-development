from typing import Any, Dict, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()


class EmotionBotState(BaseModel):
  user_message: str = Field(default="", description="사용자 입력 메시지")
  emotion: str = Field(default="", description="분석된 감정")
  response: str = Field(default="", description="최종 응답 메시지")


llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")


def analyze_emotion(state: EmotionBotState) -> Dict[str, Any]:
  message = state.user_message
  print(f"LLM 감정 분석 중: {message}")
  messages = [
    SystemMessage(
      content="당신은 감정 분석 전문가입니다. 사용자의 메세지를 분석하여 'positive', 'negative', 'neutral' 중 하나로 감정을 분류하세요. 답변은 반드시 하나의 단아만 출력하세요."
    ),
    HumanMessage(content=f"다음 메시지의 감정을 분석해주세요.: {message}"),
  ]
  response = llm.invoke(input=messages)
  emotion = response.content.strip().lower()

  if emotion not in ["positive", "negative", "neutral"]:
    emotion = "neutral"
  print(f"LLM 감정 분석 결과: {emotion}")
  return {"emotion": emotion}


def generate_positive_response(state: EmotionBotState) -> Dict[str, Any]:
  response = "당신의 긍정적인 에너지가 느껴져요! 좋은 하루 보내세요!"
  print(f"긍정적 응답 생성: {response}")
  return {"response": response}


def generate_negative_response(state: EmotionBotState) -> Dict[str, Any]:
  response = "힘든 시간을 보내고 계시군요. 필요하다면 언제든지 이야기해 주세요."
  print(f"부정적 응답 생성: {response}")
  return {"response": response}


def generate_neutral_response(state: EmotionBotState) -> Dict[str, Any]:
  response = "메시지를 잘 받았습니다. 더 이야기해 주세요."
  print(f"중립적인 응담 생성: {response}")
  return {"response": response}


def route_by_emotion(
  state: EmotionBotState,
) -> Literal["positive_response", "negative_response", "neutral_response"]:
  emotion = state.emotion
  print(f"라우팅: {emotion}")

  if emotion == "neutral":
    return "neutral_response"
  elif emotion == "positive":
    return "positive_response"
  else:
    return "negative_response"


def create_emotion_bot_graph():
  workflow = StateGraph(EmotionBotState)

  workflow.add_node("analyze_emotion_node", analyze_emotion)
  workflow.add_node("positive_response_node", generate_positive_response)
  workflow.add_node("negative_response_node", generate_negative_response)
  workflow.add_node("neutral_response_node", generate_neutral_response)

  workflow.add_edge(START, "analyze_emotion_node")
  workflow.add_conditional_edges(
    "analyze_emotion_node",
    route_by_emotion,
    {
      "positive_response": "positive_response_node",
      "negative_response": "negative_response_node",
      "neutral_response": "neutral_response_node",
    },
  )
  workflow.add_edge("positive_response_node", END)
  workflow.add_edge("negative_response_node", END)
  workflow.add_edge("neutral_response_node", END)

  return workflow.compile()


def main():
  print("=== 감정 분석 챗봇 테스트 ===\n")
  app = create_emotion_bot_graph()

  test_cases = [
    "오늘 정말 기분이 좋아요!",
    "너무 슬프고 힘들어요...",
    "날씨가 어떤가요?",
  ]

  for i, message in enumerate(test_cases, 1):
    print(f"테스트 {i}: '{message}'")
    state = EmotionBotState(user_message=message)
    result = app.invoke(state)
    print(f"응답: {result['response']}\n")

  mermaid_png = app.get_graph().draw_mermaid_png()
  with open("./02_conditional_routing.png", "wb") as f:
    f.write(mermaid_png)


if __name__ == "__main__":
  main()
