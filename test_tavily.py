import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")

if not api_key:
    raise ValueError(
        "TAVILY_API_KEY is missing from your .env file."
    )

search = TavilySearch(
    tavily_api_key=api_key,
    max_results=3,
    topic="general",
)

try:
    result = search.invoke(
        {"query": "What is LangChain in Python?"}
    )

    print("\n========== TAVILY TEST ==========")
    print("Result type:", type(result).__name__)
    print("Search result:")
    print(result)
    print("=================================")

except Exception as e:
    print("Tavily search failed.")
    print("Error type:", type(e).__name__)
    print("Error:", str(e))

