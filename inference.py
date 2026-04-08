import os
import json
from openai import OpenAI
from src.env import CodeReviewEnv, CodeReviewAction

API_BASE_URL = os.environ.get("API_BASE_URL", "<your-active-api-base>")
MODEL_NAME = os.environ.get("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN if HF_TOKEN else "dummy")

def ask_llm(broken_query, schema, hint):
    try:
        prompt = (
            "You are an expert SQL debugger. Fix the broken SQL query below.\n"
            "Schema: " + schema + "\n"
            "Hint: " + hint + "\n"
            "Broken Query: " + broken_query + "\n"
            "Reply with ONLY the fixed SQL query, nothing else."
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(json.dumps({"event": "error", "message": str(e)}), flush=True)
        return broken_query

def main():
    try:
        env = CodeReviewEnv()
        obs = env.reset()
        total_reward = 0.0
        step_num = 0

        print(json.dumps({
            "event": "[START]",
            "task_id": obs.task_id,
            "broken_query": obs.broken_query,
            "schema": obs.db_schema
        }), flush=True)

        while obs.task_id != "done":
            fixed_query = ask_llm(obs.broken_query, obs.db_schema, obs.hint)
            action = CodeReviewAction(fixed_query=fixed_query)
            result = env.step(action)
            total_reward += result["reward"]
            step_num += 1

            print(json.dumps({
                "event": "[STEP]",
                "step": step_num,
                "task_id": result["info"]["task_id"],
                "fixed_query": fixed_query,
                "reward": result["reward"],
                "cumulative_score": result["info"]["score"]
            }), flush=True)

            obs = result["observation"]

        print(json.dumps({
            "event": "[END]",
            "total_steps": step_num,
            "total_reward": total_reward,
            "final_score": total_reward / 3.0
        }), flush=True)

    except Exception as e:
        print(json.dumps({"event": "error", "message": str(e)}), flush=True)
        raise

if __name__ == "__main__":
    main()
