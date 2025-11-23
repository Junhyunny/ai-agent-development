import rich
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage

load_dotenv()

messages = [
  SystemMessage(
    content="당신은 사용자의 질문에 간결하고 명확하게 답변하는 AI 도우미입니다."
  ),
  HumanMessage(content="LangChain에 대해 설명해주세요."),
  AIMessage(
    content="LangChain은 언어 모델을 활용한 애플리케이션 개발을 용이하게 하는 프레임워크입니다. 다양한 도구와 통합하여 복잡한 작업을 자동화할 수 있습니다."
  ),
  HumanMessage(content="주요 기능 세가지만 알려주세요."),
]

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
result = model.invoke(messages)
rich.print(result.content)
