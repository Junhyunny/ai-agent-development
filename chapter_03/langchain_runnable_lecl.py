from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

prompt = ChatPromptTemplate.from_template(
  "주어지는 문구에 대하여 50자 이내의 짧은 시를 작성해주세요. : {word}"
)
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"word": "평범한 일상"})

print(result)
