import sqlite3
from openenv import core
Environment = core.Environment
Action = core.Action
Observation = core.Observation
State = core.State


class CodeReviewAction(Action):
    fixed_query: str


class CodeReviewObservation(Observation):
    task_id: str
    broken_query: str
    schema: str
    hint: str
    expected_output: str


class CodeReviewState(State):
    current_task: int
    total_tasks: int
    score: float


TASKS = [
    {
        "id": "easy_1",
        "broken_query": "SELEC * FROM employees",
        "fixed_query": "SELECT * FROM employees",
        "schema": "employees(id INTEGER, name TEXT, salary REAL, dept_id INTEGER)",
        "hint": "Check for typos in the SQL keyword",
        "expected_output": "[(1, 'Alice', 50000.0, 1), (2, 'Bob', 60000.0, 1), (3, 'Carol', 45000.0, 2)]",
        "difficulty": "easy"
    },
    {
        "id": "medium_1",
        "broken_query": "SELECT name, salary FROM employees WHERE salary > 55000 ORDER name",
        "fixed_query": "SELECT name, salary FROM employees WHERE salary > 55000 ORDER BY name",
        "schema": "employees(id INTEGER, name TEXT, salary REAL, dept_id INTEGER)",
        "hint": "ORDER clause is missing a keyword",
        "expected_output": "[('Bob', 60000.0)]",
        "difficulty": "medium"
    },
    {
        "id": "hard_1",
        "broken_query": "SELECT d.name, COUNT(e.id) FROM employees e JOIN department d ON e.dept_id = d.id GROUP e.dept_id",
        "fixed_query": "SELECT d.name, COUNT(e.id) FROM employees e JOIN department d ON e.dept_id = d.id GROUP BY e.dept_id",
        "schema": "employees(id INTEGER, name TEXT, salary REAL, dept_id INTEGER), department(id INTEGER, name TEXT)",
        "hint": "GROUP clause is missing a keyword",
        "expected_output": "[('Engineering', 2), ('HR', 1)]",
        "difficulty": "hard"
    }
]


class CodeReviewEnv(Environment):
    def __init__(self):
        self.current_task = 0
        self.score = 0.0
        self.db = self._setup_db()

    def _setup_db(self):
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        cur.execute("CREATE TABLE employees (id INTEGER, name TEXT, salary REAL, dept_id INTEGER)")
        cur.execute("INSERT INTO employees VALUES (1, 'Alice', 50000.0, 1)")
        cur.execute("INSERT INTO employees VALUES (2, 'Bob', 60000.0, 1)")
        cur.execute("INSERT INTO employees VALUES (3, 'Carol', 45000.0, 2)")
        cur.execute("CREATE TABLE department (id INTEGER, name TEXT)")
        cur.execute("INSERT INTO department VALUES (1, 'Engineering')")
        cur.execute("INSERT INTO department VALUES (2, 'HR')")
        conn.commit()
        return conn

    def reset(self, **kwargs):
        self.current_task = 0
        self.score = 0.0
        self.db = self._setup_db()
        task = TASKS[self.current_task]
        return CodeReviewObservation(
            task_id=task["id"],
            broken_query=task["broken_query"],
            schema=task["schema"],
            hint=task["hint"],
            expected_output=task["expected_output"]
        )

    def step(self, action):
        task = TASKS[self.current_task]
        reward = self._grade(action.fixed_query, task)
        self.score += reward
        self.current_task += 1
        done = self.current_task >= len(TASKS)
        if done:
            obs = CodeReviewObservation(
                task_id="done",
                broken_query="",
                schema="",
                hint="All tasks completed!",
                expected_output=""
            )
        else:
            next_task = TASKS[self.current_task]
            obs = CodeReviewObservation(
                task_id=next_task["id"],
                broken_query=next_task["broken_query"],
                schema=next_task["schema"],
                hint=next_task["hint"],
                expected_output=next_task["expected_output"]
            )
        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": {"task_id": task["id"], "score": self.score}
        }

    def _grade(self, fixed_query, task):
        try:
            cur = self.db.cursor()
            cur.execute(fixed_query)
            result = str(cur.fetchall())
            if result == task["expected_output"]:
                return 1.0
            elif fixed_query.strip().upper() != task["broken_query"].strip().upper():
                return 0.3
            return 0.0
        except Exception:
            return 0.0

    def state(self):
        return CodeReviewState(
            current_task=self.current_task,
            total_tasks=len(TASKS),
            score=self.score
        )
