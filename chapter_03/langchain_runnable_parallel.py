import rich
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

load_dotenv()

prompt = ChatPromptTemplate.from_template(
  "주어진 '{word}'와 유사한 단어 3가지를 나열해주세요. 단어만 나열합니다."
)
llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")
parser = StrOutputParser()

analysis_chain = RunnableParallel(
  synonyms=prompt | llm | parser,
  word_count=RunnableLambda(lambda x: len(x["word"])),
  uppercase=RunnableLambda(lambda x: x["word"].upper()),
)

result = analysis_chain.invoke({"word": "peaceful"})

rich.print(result)
