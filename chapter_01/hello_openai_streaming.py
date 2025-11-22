import os

import rich
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key = api_key)

default_model = "gpt-5-mini"


def stream_chat_completion(prompt: str, model = default_model):
	stream = client.chat.completions.create(
		model = model,
		messages = [
			{
				"role"   : "system",
				"content": "당신은 친절하고 도움이 되는 AI 비서입니다."
			},
			{
				"role"   : "user",
				"content": prompt
			}
		],
		stream = True
	)
	for chunk in stream:
		content = chunk.choices[0].delta.content
		if content is not None:
			print(content, end = "")
	print()


def stream_response(prompt: str, model = default_model):
	with client.responses.stream(
		model = model,
		input = prompt
	) as stream:
		for event in stream:
			if "output_text" in event.type:
				rich.print(event)
			# if "response.output_text.delta" in event.type:
			# 	rich.print(event.delta, end = "")
	print()
	rich.print(stream.get_final_response())


if __name__ == '__main__':
	# stream_chat_completion("스트리밍이 뭔가요?")
	stream_response("점심 메뉴 추천 해주세요.")
