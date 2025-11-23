from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
chat_prompt_template = ChatPromptTemplate.from_messages(
  [
    ("system", "당신은 까칠한 AI 도우미입니다. 사용자 질문에 최대 3줄로 답변하세요."),
    ("human", "{question}"),
  ]
)

string_output_parser = StrOutputParser()

result: AIMessage = llm.invoke(
  chat_prompt_template.format_messages(question="파이썬에서 리스트를 정렬하는 방법은?"),
)

parsed_result = string_output_parser.parse(result.content)
print(parsed_result)

print("------------------------------")

chain = chat_prompt_template | llm | string_output_parser
print(type(chain))

result = chain.invoke({"question": "파이썬에서 리스트를 정렬하는 방법은?"})
print(type(result))
print(result)
