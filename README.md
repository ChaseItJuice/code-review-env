\# Code Review Environment



An OpenEnv-compatible RL environment where an AI agent debugs and fixes broken SQL queries.



\## Description



The agent receives broken SQL queries one at a time and must return the corrected version.

Tasks increase in difficulty from easy → medium → hard.



\## Action Space



| Field | Type | Description |

|-------|------|-------------|

| fixed\_query | string | The corrected SQL query |



\## Observation Space



| Field | Type | Description |

|-------|------|-------------|

| task\_id | string | Unique identifier for the task |

| broken\_query | string | The broken SQL query to fix |

| schema | string | The database schema context |

| hint | string | A hint about what is wrong |

| expected\_output | string | Expected result after fix |



\## Tasks



| ID | Difficulty | Description |

|----|------------|-------------|

| easy\_1 | Easy | Fix a typo in a basic SELECT statement |

| medium\_1 | Medium | Fix a missing keyword in ORDER BY clause |

| hard\_1 | Hard | Fix a missing keyword in GROUP BY with JOIN |



\## Reward



\- `1.0` — Correct fix, query produces expected output

\- `0.3` — Partial credit, query changed but output incorrect

\- `0.0` — No change or query throws an error



\## Setup

```bash

pip install -r requirements.txt

```



\## Running Inference

```bash

export API\_BASE\_URL=https://api.groq.com/openai/v1

export MODEL\_NAME=llama-3.3-70b-versatile

export HF\_TOKEN=your\_token\_here

python inference.py

```



\## Environment Variables



| Variable | Description |

|----------|-------------|

| API\_BASE\_URL | The LLM API endpoint |

| MODEL\_NAME | Model identifier for inference |

| HF\_TOKEN | Your Hugging Face / API key |

