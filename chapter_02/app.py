import uvicorn
from fastapi import FastAPI, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse

from chatbot import ChatBot

app = FastAPI()
chatbot = ChatBot()


@app.get("/", response_class=HTMLResponse)
async def root():
  chat_history = ""
  messages = chatbot.get_history_messages()
  for msg in messages:
    if msg["role"] == "user":
      chat_history += f"<p><b>당신:</b> {msg['content']}</p>"
    else:
      chat_history += f"<p><b>어린왕자:</b> {msg['content']}</p>"
  html_content = f"""
		<html>
			<body>
				<h1>어린 왕자 챗봇</h1>
				<div>{chat_history}</div>
				<form action="/chat" method="post">
					<input type="text" name="message" placeholder="메시지를 입력하세요" required>
					<button type="submit">전송</button>
				</form>
			</body>
		</html>
	"""
  return html_content


@app.post("/chat", response_class=RedirectResponse)
async def chat(message: str = Form(...)):
  chatbot.chatbot_response(input=message)
  return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)
