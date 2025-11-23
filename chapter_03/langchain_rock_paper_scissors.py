import random

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()


@tool
def rps():
  """Return a random rock, paper, or scissors choice."""
  return random.choice(["바위", "보", "가위"])


llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai").bind_tools(
  [rps]
)
llm_for_chat = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
# print(type(llm))  # langchain_core.runnables.base.RunnableBinding


def judge(user_choice, computer_choice):
  user_choice = user_choice.strip()
  computer_choice = computer_choice.strip()
  if user_choice == computer_choice:
    return "무승부"
  elif (user_choice, computer_choice) in [
    ("가위", "보"),
    ("바위", "가위"),
    ("보", "바위"),
  ]:
    return "사용자 승리"
  else:
    return "컴퓨터 승리"


print("가위!바위!보! (종료: q)")
while (user_input := input("\n가위/바위/보: ")) != "q":
  ai_msg = llm.invoke(
    f"가위바위보 게임: 사용자가 {user_input}을 냈습니다. rps tool을 사용하세요."
  )
  if (
    ai_msg.tool_calls
  ):  # ai_message 에 어떤 도구를 어떤 파라미터로 호출해야 하는지 정해져있다..
    print(ai_msg)
    # print(type(rps))  # langchain_core.tools.structured.StructuredTool
    llm_choice = rps.invoke(
      ""
    )  # ai_msg.tool_calls 를 사용해서 해당 도구를 호출할 수 있다 이 예제에선 명시적으로
    print(f"LLM이 선택한 도구: {llm_choice}")
    result = judge(user_input, llm_choice)
    print(f"승부: {result}")
    final = llm_for_chat.invoke(
      f"가위바위보 게임 결과를 재미있게 해설해주세요. 사용자: {user_input}, AI: {llm_choice}, 결과: {result}"
    )
    print(final)
    print(f"LLM 해설: {final.content}")
    print(f"게임 요약. 사용자: {user_input}, AI: {llm_choice}, 결과: {result}")
  else:
    print("LLM이 도구를 호출하지 않았습니다. 다시 시도해주세요.")
