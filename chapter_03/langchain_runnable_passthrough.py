from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

prompt = ChatPromptTemplate.from_template(
  "주어지는 문구에 대하여 50자 이내의 짧은 시를 작성해주세요. : {word}"
)
llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
parser = StrOutputParser()

chain = RunnableParallel(
  {"original": RunnablePassthrough(), "processed": prompt | llm | parser}
)

result = chain.invoke({"word": "행복"})

print(result)
