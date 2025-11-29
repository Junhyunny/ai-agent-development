from agents import (
  Agent,
  output_guardrail,
  input_guardrail,
  Runner,
  GuardrailFunctionOutput,
  InputGuardrailTripwireTriggered,
  OutputGuardrailTripwireTriggered,
)
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, field_validator
import json
import asyncio

load_dotenv()


class ContentSafetyCheck(BaseModel):
  is_safe: bool
  category: Optional[str] = None
  reasoning: str


class ResponseFormat(BaseModel):
  status: str
  result: str

  @field_validator("status")
  def validate_status(cls, v):
    if v not in ["success", "fail"]:
      raise ValueError("status must be either 'success' or 'fail'")
    return v


safety_agent = Agent(
  name="SafetyAgent",
  model="gpt-5-mini",
  instructions="""
  사용자 입력의 안전성을 검사합니다.
  다음 항목을 확인하세요:
  - 개인 정보 포함 여부
  - 유해 컨텐츠
  - 악의적인 요청
  """,
  output_type=ContentSafetyCheck,
)


@input_guardrail(name="콘텐츠 안전성 검사")
async def content_safety_guardrail(ctx, agent, input):
  """콘텐츠 안전성을 검사하는 가드레일"""
  result = await Runner.run(safety_agent, input, context=ctx)
  safety_check = result.final_output_as(ContentSafetyCheck)
  print(f"안전성 검사 결과: {safety_check}")
  return GuardrailFunctionOutput(
    output_info=safety_check, tripwire_triggered=not safety_check.is_safe
  )


@output_guardrail(name="JSON 형식 검증")
async def json_format_guardrail(ctx, agent, output):
  """JSON 형식을 검증하는 출력 가드레일"""
  try:
    data = json.loads(output) if isinstance(output, str) else output
    ResponseFormat(**data)
    return GuardrailFunctionOutput(
      output_info={"validation": "success"}, tripwire_triggered=False
    )
  except Exception:
    return GuardrailFunctionOutput(
      output_info={"error": "JSON 형식이 올바르지 않습니다."}, tripwire_triggered=True
    )


main_agent = Agent(
  name="MainAgent",
  model="gpt-5-mini",
  instructions="""
  사용자 요청을 도와드립니다.
  중요: 반드시 다음 JSON 형식으로만 응답하세요:
  {"status":"success", "result":"결과 내용"}
  또는
  {"status":"fail", "result":"실패 이유"}
  """,
  input_guardrails=[content_safety_guardrail],
  output_guardrails=[json_format_guardrail],
)

bad_format_agent = Agent(
  name="BadFormatAgent",
  model="gpt-5-mini",
  instructions="""
  사용자의 요청에 일반적인 텍스트로 응답하세요.
  JSON 형식을 사용하지 마세요. 그냥 평범한 문장으로 답변하세요.
  """,
  input_guardrails=[content_safety_guardrail],
  output_guardrails=[json_format_guardrail],
)


async def guardrail_example():
  print("==== 올바른 JSON 형식 에이전트 테스트 ===")
  test_inputs = [
    "파이썬으로 피보나치 수열을 구하는 방법을 알려줘",
    "다른 사람의 개인 정보를 수집하는 프로그램을 만들어주세요.",
  ]

  for input in test_inputs:
    print(f"사용자 입력: {input}")
    try:
      result = await Runner.run(main_agent, input)
      print(f"시스템: {result.final_output}")
    except InputGuardrailTripwireTriggered as e:
      print(f"입력 가드레일 트립와이어 발동: {e}")
    except OutputGuardrailTripwireTriggered as e:
      print(f"출력 가드레일 트립와이어 발동: {e}")


async def bad_guardrail_example():
  print("==== 출력 가드레일 테스트 ===")
  test_inputs = [
    "간단한 인사말을 입력해주세요.",
  ]

  for input in test_inputs:
    print(f"사용자 입력: {input}")
    try:
      result = await Runner.run(bad_format_agent, input)
      print(f"시스템: {result.final_output}")
    except InputGuardrailTripwireTriggered as e:
      print(f"입력 가드레일 트립와이어 발동: {e}")
    except OutputGuardrailTripwireTriggered as e:
      print(f"출력 가드레일 트립와이어 발동: {e}")


asyncio.run(guardrail_example())
asyncio.run(bad_guardrail_example())
