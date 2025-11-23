import rich
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch

load_dotenv()


def is_english(x: dict) -> bool:
  return all(ord(char) < 128 for char in x["word"])


ko_prompt = ChatPromptTemplate.from_template(
  "주어진 '{word}'와 유사한 단어 3가지를 나열해주세요. 단어만 나열합니다."
)
en_prompt = ChatPromptTemplate.from_template(
  "Give me 3 synonyms for the word '{word}'. List only the words."
)
llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
parser = StrOutputParser()

chain = RunnableBranch(
  (is_english, en_prompt | llm | parser),
  ko_prompt | llm | parser,
)

result = chain.invoke({"word": "peaceful"})
rich.print(result)

result = chain.invoke({"word": "행복"})
rich.print(result)
