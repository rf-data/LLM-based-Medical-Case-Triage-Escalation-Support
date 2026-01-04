import os
from openai import OpenAI

import src.utils.general_helper as gh

gh.load_env_vars()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=api_key)

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say OK"}],
)

print(resp.choices[0].message.content)
