from fastapi import FastAPI
from src.env import CodeReviewEnv, CodeReviewAction

app = FastAPI()
env = CodeReviewEnv()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(action: CodeReviewAction):
    result = env.step(action)
    result["observation"] = result["observation"].dict()
    return result

@app.get("/state")
def state():
    return env.state().dict()