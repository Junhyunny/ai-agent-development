import asyncio
import logging
import os
import random

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

client = AsyncOpenAI(api_key = os.getenv("OPENAI_API_KEY"))


async def simulate_random_failure():
	if random.random() < 0.5:
		logger.warning("Simulating failure")
		raise Exception("Simulated failure")
	await asyncio.sleep(random.uniform(0.1, 0.5))


@retry(
	stop = stop_after_attempt(3),
	wait = wait_exponential(multiplier = 1, min = 2, max = 10),  # 지수 백오프: 2초, 4초, 8초
	retry = retry_if_exception_type(),
	before_sleep = lambda retry_state: logger.warning(f"API failure: {retry_state.outcome.exception()}, {retry_state.attempt_number}")
)
async def call_async_openai(prompt: str, model: str = "gpt-5-mini"):
	logger.info(f"OpenAI API 호출")
	await simulate_random_failure()
	response = await client.chat.completions.create(
		model = model,
		messages = [{
			"role"   : "user",
			"content": prompt
		}]
	)
	logger.info(f"OpenAI API 응답 완료")
	return response.choices[0].message.content


async def main():
	print("API 호출")
	prompt = "비동기 프로그래밍에 대해 두세 문장으로 설명해주세요."

	task = call_async_openai(prompt)

	try:
		response = await asyncio.gather(task)
		print(response)
	except Exception as e:
		logger.error(f"API 호출 중 처리되지 않은 오류 발생: {e}")


if __name__ == "__main__":
	asyncio.run(main())
