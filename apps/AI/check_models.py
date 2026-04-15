import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import requests

key = os.getenv("GROQ_API_KEY")
resp = requests.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {key}"}
)
models = resp.json().get("data", [])
print("Available Groq models:")
for m in sorted(models, key=lambda x: x["id"]):
    print(f"  - {m['id']}")
