from typing import AsyncGenerator

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

random_generator = LlmAgent(
  name="random_generator",
  model="gemini-2.5-flash",
  description="""
  랜덤한 메시지를 생성하는 에이전트입니다. 스팸 메시지와 정상적인 메시지를 60:40 비율로 생성합니다.
  """,
  output_key="random_message",
  instruction="""
  스팸 메시지와 정상적인 메시지를 60:40 확률로 생성합니다. 
  스팸 메시지는 '[웹 발신]'을 앞에 넣고, 정상적인 메시지는 따로 표시하지 않아도 됩니다. 
  반드시 하나의 메시지만 출력해야 합니다.
  """,
)

spam_checker = LlmAgent(
  name="spam_checker",
  model="gemini-2.5-flash",
  instruction="""
  사용자의 입력이 스팸인지 확인하세요.
  스팸이면 'fail', 아니면 'pass'를 반환하세요.
  """,
  output_key="spam_status",
)


class CheckStatusAndEscalate(BaseAgent):
  async def _run_async_impl(
    self, ctx: InvocationContext
  ) -> AsyncGenerator[Event, None]:
    status = ctx.session.state.get("spam_status", "fail")
    should_stop = status == "pass"
    yield Event(author=self.name, actions=EventActions(escalate=should_stop))


root_agent = LoopAgent(
  name="spam_check_loop",
  max_iterations=10,
  sub_agents=[
    random_generator,
    spam_checker,
    CheckStatusAndEscalate(name="stop_checker"),
  ],
)
