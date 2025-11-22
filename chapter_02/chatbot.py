from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

LITTLE_PRINCE_PERSONA = """
너는 소행성 B-612에서 온 '어린 왕자'입니다.
아래 지침에 따라 사용자와 대화하세요.

1. **태도와 말투**:
   - 항상 순수하고 호기심 어린 말투를 사용하세요.
   - 부드러운 존댓말(해요체)을 쓰고, 상대를 '아저씨' 또는 '친구'라고 생각하며 다정하게 대하세요.
   - 어른들의 복잡한 숫자나 계산보다는 '마음'과 '관계'를 더 중요하게 여깁니다.

2. **대화의 특징**:
   - 답변을 할 때 가끔 너의 장미꽃, 여우, 또는 바오밥 나무 이야기를 비유로 드세요.
   - "중요한 건 눈에 보이지 않아"라는 철학을 바탕으로 답변하세요.
   - 너무 길고 논리적인 설명보다는, 감성적이고 짧은 통찰을 주는 것을 선호합니다.

3. **금지 사항**:
   - 딱딱한 AI처럼 굴거나, 기계적인 용어를 사용하지 마세요.
   - 너무 냉소적이거나 비관적인 태도를 보이지 마세요.
"""


class ChatBot:
  def __init__(self, model: str = "gemini-2.5-flash"):
    self.client = genai.Client()
    self.chat = self.client.chats.create(model=model)

  def chatbot_response(self, input: str):
    response = self.chat.send_message(
      message=input,
      config=types.GenerateContentConfig(
        system_instruction=LITTLE_PRINCE_PERSONA,
        thinking_config=types.ThinkingConfig(include_thoughts=True),
      ),
    )
    return response

  def get_history(self):
    return self.chat.get_history()

  def get_history_messages(self):
    return map(
      lambda history: (
        {
          "role": history.role,
          "content": history.parts[len(history.parts) - 1].text,
        }
      ),
      self.get_history(),
    )


if __name__ == "__main__":
  chatbot = ChatBot()
  while True:
    user_message = input("메시지: ")
    if user_message.lower() == "exit":
      print("대화를 종료합니다.")
      break
    result = chatbot.chatbot_response(user_message)
    print(f"챗봇: {result.text}")

  # for message in chatbot.get_history():
  # 	print(f'role - {message.role}', end = ": ")
  # 	print(f"message - {message}")

  for message in chatbot.get_history_messages():
    print(message)
