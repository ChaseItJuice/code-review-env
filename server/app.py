from fastapi import FastAPI
from env import CodeReviewEnv

app = FastAPI()
env = CodeReviewEnv()

@app.get("/")
def read_root():
    return {"status": "ok", "env": "code-review-env"}

@app.post("/reset")
def reset():
    obs, info = env.reset()
    return {"observation": obs.__dict__, "info": info}

@app.post("/step")
def step(action: dict):
    obs, reward, terminated, truncated, info = env.step(action.get("action", ""))
    return {
        "observation": obs.__dict__,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info
    }