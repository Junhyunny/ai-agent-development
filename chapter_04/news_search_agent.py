import asyncio

from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
from duckduckgo_search import DDGS

load_dotenv()


@function_tool()
def news_search(query: str):
  """덕덕고를 사용한 뉴스 검색 핸들러 함수"""
  try:
    results = DDGS().text(query, max_results=5)
    if results:
      return f"'{query}' 검색 결과: \n{results}"
    else:
      return "검색 결과가 없습니다."
  except Exception as e:
    return f"검색 중 오류가 발생했습니다: {str(e)}"


news_agent = Agent(
  name="NewsAgent",
  model="gpt-5-mini",
  instructions=(
    "당신은 한국어 뉴스 리포트입니다."
    "WebSearchTool을 사용하여 최신 뉴스를 검색하고, "
    "3개의 기사 URL을 함께 알려주세요."
  ),
  tools=[news_search],
)

if __name__ == "__main__":
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  print("뉴스 검색 에이전트를 시작합니다.")
  result = Runner.run_sync(
    starting_agent=news_agent, input="인공지능 관련 최신 기술 뉴스를 검색해주세요."
  )
  print(result.final_output)
