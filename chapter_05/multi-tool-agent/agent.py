import httpx
from google.adk.agents import Agent
from geopy.geocoders import Nominatim


def get_coordinates(city_name: str) -> tuple[float, float]:
  """도시 이름을 받아 위도와 경도를 반환합니다."""
  geolocator = Nominatim(user_agent="weather_app")
  location = geolocator.geocode(city_name)
  if location:
    return location.latitude, location.longitude
  else:
    raise ValueError(f"could not find coordinates for {city_name}")


def get_weather(city_name: str) -> dict:
  """도시 이름을 받아 해당 도시의 현재 날씨 정보를 반환한다."""
  if city_name:
    latitude, longitude = get_coordinates(city_name)
  else:
    raise ValueError("city name must be provided to get weather information")

  url = f"http://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
  response = httpx.get(url)
  response.raise_for_status()
  return response.json()


def get_kbo_rang() -> dict:
  """한국 프로야구 구단의 랭킹을 가져오는 함수입니다."""
  response = httpx.get(
    "https://sports.daum.net/prx/hermes/api/team/rank.json?leagueCode=kbo&seasonKey=2025"
  )
  return response.json()


root_agent = Agent(
  name="weather_agent",
  model="gemini-2.5-flash",
  description="날씨 정보와 KBO 랭킹을 제공하는 에이전트입니다.",
  instruction="도시 이름을 입력하면 해당 도시의 날씨 정보를 제공하고, 'KBO 랭킹'이라고 입력하면 한국 프로야구 구단의 랭킹을 제공합니다.",
  tools=[get_weather, get_kbo_rang],
)
