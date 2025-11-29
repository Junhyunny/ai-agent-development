import asyncio

from dotenv import load_dotenv
from agents import Agent, Runner

load_dotenv()


async def simple_handoff_example():
  print("Agent 병원 안내 시스템\n")
  print("=" * 50)
  doctor_agent = Agent(
    name="DoctorAgent",
    instructions="근골격계 문제(허리 통증, 관전염, 골절 등)를 진료합니다.",
  )
  second_doctor_agent = Agent(
    name="SecondDoctorAgent",
    instructions="내과 질환(감기, 소화불량, 두통 등)을 진료합니다. 근골격계 문제는 정형외가 의사에게 연결합니다.",
    handoffs=[doctor_agent],
  )
  receptionist_agent = Agent(
    name="ReceptionistAgent",
    instructions="""
    환자의 증상을 듣고 적절한 의사에게 연결해줍니다.
    - 감기, 소화불량, 두통: SecondDoctorAgent
    - 허리, 관절, 골절: DoctorAgent
    """,
    handoffs=[second_doctor_agent, doctor_agent],
  )

  response_id = None
  current_agent = receptionist_agent
  conversations = [
    "안녕하세요, 며칠 전부터 머리가 아파요.",
    "커피를 마시면 아파요. 허리도 아파요.",
    "운동을 하면 좋아질까요?",
  ]
  for msg in conversations:
    print(f"\n 환자: {msg}")
    if response_id:
      result = await Runner.run(current_agent, msg, previous_response_id=response_id)
    else:
      result = await Runner.run(current_agent, msg)
    response_id = result.last_response_id
    if current_agent != result.last_agent:
      print(
        f"<핸드오프 발생> {current_agent.name}에서 {result.last_agent.name}로 핸드오프"
      )
      current_agent = result.last_agent
    print(f" 에이전트 병원: {current_agent.name}: {result.final_output}")


if __name__ == "__main__":
  asyncio.run(simple_handoff_example())
