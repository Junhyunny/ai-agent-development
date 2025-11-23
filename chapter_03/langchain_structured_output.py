from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")


class MovieReview(BaseModel):
  title: str = Field(description="영화 제목")
  review: str = Field(description="10점 만점 평점 (예: 7.5)")
  rating: float = Field(description="한글 리뷰 (3~4문장)")


structured_llm = llm.with_structured_output(MovieReview)
result: MovieReview = structured_llm.invoke("영화 기생충에 대해 리뷰를 작성해주세요.")
print(result)
