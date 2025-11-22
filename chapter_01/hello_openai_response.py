import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key = api_key)


def get_responses(message: str, model = "gpt-5-mini") -> str:
	response = client.responses.create(
		model = model,
		tools = [{
			"type": "web_search_preview"
		}],
		input = message
	)
	return response.output_text


if __name__ == '__main__':
	prompt = """
		https://platform.openai.com/docs/api-reference/responses/create 를 읽어서 리스폰스 API에 대해 요약해주세요.
	"""
	output = get_responses(prompt)
	print(f"\noutput: {output}")
