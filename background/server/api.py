from fastapi import FastAPI
from background.QA import answer

app = FastAPI()


@app.get("/answer/{item}")
async def send_answer(item):
    result = await answer(item)
    return {"answer": result}


