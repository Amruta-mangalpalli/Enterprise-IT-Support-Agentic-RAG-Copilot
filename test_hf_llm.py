import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY is missing. Check your project's .env file."
    )

chat = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=api_key,
)

response = chat.invoke(
    "Explain Agentic RAG in two sentences."
)

print("\nGroq LLM response:")
print(response.content)

