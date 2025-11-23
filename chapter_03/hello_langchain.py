import rich
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
result = model.invoke("랭체인이 뭔가요?")
rich.print(
  result.content
)  # 응답은 AIMesage 타입이며 content 변수에 실제 결과 값이 저장되어 있고, 해당 값을 출력한다.
