import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)


def get_chat_completion(message: str, model="gpt-5-mini") -> str:
  response = client.chat.completions.create(
    model=model,
    messages=[
      {"role": "system", "content": "당신은 친절하고 도움이 되는 AI 비서입니다."},
      {"role": "user", "content": message},
    ],
  )
  return response.choices[0].message.content


if __name__ == "__main__":
  user_prompt = input("AI에게 물어볼 질문을 입력하세요.\n")
  response = get_chat_completion(user_prompt)
  print(f"\nfrom AI: {response}")
