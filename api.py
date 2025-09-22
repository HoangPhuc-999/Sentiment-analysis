from fastapi import FastAPI, Form
import uvicorn

from model import LogisticRegressionModel
from helper import get_llm, classification_modeL_cache, llm_model_cache, prompt

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Sentiment Analysis API is running."}

@app.post("/chat", response_model=str)
async def chat_endpoint(message: str = Form(...)):
    if "model" not in classification_modeL_cache:
        classification_modeL_cache["model"] = LogisticRegressionModel()
    
    if "llm" not in llm_model_cache:
        llm_model_cache["llm"] = get_llm()

    prediction = classification_modeL_cache["model"].predict(message)
    sentiment = "Positive" if prediction[0][1] > 0.5 else "Negative"

    result = llm_model_cache["llm"].invoke(
        prompt.format(text=message, prediction=sentiment))

    if result:
        return result.content

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=7861)