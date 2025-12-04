from google.adk.agents import Agent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search

weather_fetcher = Agent(
  name="weather_fetcher",
  model="gemini-2.5-flash",
  output_key="weather_info",
  instruction="오늘의 날씨 정보를 제공하세요.",
  tools=[google_search],
)

news_fetcher = Agent(
  name="news_fetcher",
  model="gemini-2.5-flash",
  output_key="news_info",
  instruction="오늘의 주요 뉴스를 요약하세요.",
  tools=[google_search],
)

stock_fetcher = Agent(
  name="stock_fetcher",
  model="gemini-2.5-flash",
  output_key="stock_info",
  instruction="주요 주식 시장 동향을 제공하세요.",
  tools=[google_search],
)

parallel_fetcher = ParallelAgent(
  name="daily_briefing",
  sub_agents=[weather_fetcher, news_fetcher, stock_fetcher],
  description="여러 정보를 동시에 수집",
)

summarizer = Agent(
  name="daily_briefing",
  model="gemini-2.5-flash",
  instruction="""
  수집된 정보를 종합하여 일일 브리핑을 작성하세요:
  - 날씨: {weather_info}
  - 뉴스: {news_info}
  - 주식: {stock_info}
  최종 브리핑을 간결하고 명확하게 작성하세요.
  """,
)

root_agent = SequentialAgent(
  name="daily_briefing_system",
  sub_agents=[parallel_fetcher, summarizer],
  description="정보를 별롤로 수집한 후 종합 브리핑 생성",
)
