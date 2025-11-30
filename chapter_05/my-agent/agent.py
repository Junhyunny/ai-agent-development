from google.adk.agents import Agent


def greet_user() -> str:
  return "Hello World!"


root_agent = Agent(
  name="hello_agent",
  model="gemini-2.5-flash",
  description="유저와 인사하는 에이전트입니다.",
  instruction="사용자에게 반갑고 친절하게 인사해주세요.",
  tools=[greet_user],
)
