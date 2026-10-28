import json
import logging
from typing import Literal

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END

from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever


logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None
_web_search = None

load_dotenv()


def llm():
    """Initialize and cache the Groq chat model."""

    global _llm

    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing from your .env file."
            )

        _llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0,
            api_key=api_key,
        )

    return _llm

def web_search_tool():
    """Initialize and cache the Tavily search tool."""

    global _web_search

    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is missing from your .env file."
            )

        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )

    return _web_search


def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]

def parse_json_response(response, expected_field: str):
    """
    Parse a JSON object from model output.

    The model must return an object containing the expected field.
    """

    content = getattr(response, "content", response)

    if not isinstance(content, str):
        content = str(content)

    content = content.strip()

    # Handle JSON enclosed in Markdown fences.
    if content.startswith("```"):
        lines = content.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        content = "\n".join(lines).strip()

    # Parse the response as a JSON object.
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end <= start:
            raise ValueError(
                f"Model did not return valid JSON: {content}"
            )

        data = json.loads(content[start:end + 1])

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from the model.")

    if expected_field not in data:
        raise ValueError(
            f"JSON response is missing '{expected_field}'."
        )

    return data


def invoke_json(prompt: str, expected_field: str):
    """Invoke the model and parse its JSON response."""

    response = llm().invoke([HumanMessage(content=prompt)])
    return parse_json_response(response, expected_field)


def route_question(state: AgentState):
    prompt = f"""
You route questions for an enterprise IT support assistant.

Use "kb" for questions about:
- Company IT policies
- Password resets and MFA
- VPN and email
- Laptop setup and troubleshooting
- Software access
- Security and company devices
- Technical support

Use "direct" only for greetings, thanks, and casual chat
that does not require company knowledge.

Return only a JSON object:
{{"route": "kb"}}

The route must be either "kb" or "direct".

Question:
{state["question"]}
"""

    try:
        data = invoke_json(prompt, "route")
        route = data["route"]

        if route not in {"kb", "direct"}:
            raise ValueError(f"Invalid route: {route}")

    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Router output invalid: %s", exc)

        # Fail safely: retrieve company knowledge rather than
        # assuming an IT question is casual chat.
        route = "kb"

    decision = RouteDecision(route=route)

    return {
        "source_used": decision.route,
        "trace": add_trace(
            state,
            f"Router → {decision.route.upper()}",
        ),
    }


def route_after_router(
    state: AgentState,
) -> Literal["retrieve_kb", "direct_answer"]:

    return (
        "retrieve_kb"
        if state["source_used"] == "kb"
        else "direct_answer"
    )

def retrieve_kb(state: AgentState):
    docs = get_retriever().invoke(state["current_query"])

    return {
        "kb_docs": docs,
        "trace": add_trace(
            state,
            f"Private KB retrieval → {len(docs)} chunks",
        ),
    }

def grade_kb(state: AgentState):
    context = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n"
        f"{doc.page_content}"
        for doc in state["kb_docs"]
    )

    prompt = f"""
Evaluate whether the private company knowledge base
contains sufficient evidence to answer the question.

Return only a JSON object:
{{"grade": "good"}}

Use "good" only when the evidence directly and sufficiently
answers the question. Otherwise use "weak".

Question:
{state["question"]}

Private company knowledge base:
{context}
"""

    try:
        data = invoke_json(prompt, "grade")
        grade = data["grade"]

        if grade not in {"good", "weak"}:
            raise ValueError(f"Invalid evidence grade: {grade}")

    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("KB grading output invalid: %s", exc)
        grade = "weak"

    evidence_grade = EvidenceGrade(grade=grade)

    return {
        "kb_grade": evidence_grade.grade,
        "trace": add_trace(
            state,
            f"KB evidence grade → {evidence_grade.grade.upper()}",
        ),
    }


def after_kb(
    state: AgentState,
) -> Literal["generate_from_kb", "search_web"]:

    return (
        "generate_from_kb"
        if state["kb_grade"] == "good"
        else "search_web"
    )

def search_web(state: AgentState):
    result = web_search_tool().invoke(
        {"query": state["current_query"]}
    )

    lines = []
    citations = []

    if isinstance(result, dict):
        answer = result.get("answer")

        if answer:
            lines.append(f"Search answer: {answer}")

        for item in result.get("results", []):
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")

            lines.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Content: {content}"
            )

            if url:
                citations.append(
                    {
                        "title": title or url,
                        "url": url,
                        "type": "web",
                    }
                )

    else:
        lines.append(str(result))

    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(
            state,
            "Web fallback → Tavily search",
        ),
    }

def grade_web(state: AgentState):
    prompt = f"""
Evaluate whether the external web evidence directly and
sufficiently answers the question.

Return only a JSON object:
{{"grade": "good"}}

Use "good" only when the evidence is relevant and sufficient.
Otherwise use "weak".

Question:
{state["question"]}

Web evidence:
{state["web_results"]}
"""

    try:
        data = invoke_json(prompt, "grade")
        grade = data["grade"]

        if grade not in {"good", "weak"}:
            raise ValueError(f"Invalid web evidence grade: {grade}")

    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("Web grading output invalid: %s", exc)
        grade = "weak"

    evidence_grade = EvidenceGrade(grade=grade)

    return {
        "web_grade": evidence_grade.grade,
        "trace": add_trace(
            state,
            f"Web evidence grade → {evidence_grade.grade.upper()}",
        ),
    }


def after_web(
    state: AgentState,
) -> Literal[
    "generate_from_web",
    "rewrite_query",
    "insufficient",
]:

    if state["web_grade"] == "good":
        return "generate_from_web"

    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"

    return "insufficient"

def rewrite_query(state: AgentState):
    prompt = f"""
Rewrite this IT support question for improved knowledge-base
retrieval and external web search.

Preserve the original intent.
Add useful technical keywords.
Do not answer the question.
Return only the rewritten query.

Question:
{state["question"]}

Previous query:
{state["current_query"]}
"""

    response = llm().invoke([HumanMessage(content=prompt)])
    rewritten = response.content.strip()

    if not rewritten:
        rewritten = state["question"]

    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(
            state,
            f"Query rewrite → {rewritten}",
        ),
    }

def generate_from_kb(state: AgentState):
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n"
        f"{doc.page_content}"
        for doc in state["kb_docs"]
    )

    prompt = f"""
You are an enterprise IT support copilot.

Answer using ONLY the private company knowledge base below.

Requirements:
- Be concise and actionable.
- Present troubleshooting steps clearly.
- Do not invent company policies or technical procedures.
- If the evidence does not answer part of the question,
  clearly state that limitation.
- Mention that the answer is based on the private company
  knowledge base.

Question:
{state["question"]}

Private company knowledge base:
{context}
"""

    response = llm().invoke([HumanMessage(content=prompt)])
    answer = response.content.strip()

    citations = []
    seen = set()

    for doc in state["kb_docs"]:
        source = doc.metadata.get("source", "Private KB")

        if source not in seen:
            seen.add(source)

            citations.append(
                {
                    "title": source.split("/")[-1],
                    "url": "",
                    "type": "private_kb",
                }
            )

    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(
            state,
            "Answer generation → PRIVATE KB",
        ),
    }

def generate_from_web(state: AgentState):
    prompt = f"""
You are an enterprise IT support copilot.

Answer using ONLY the external web evidence below.

Requirements:
- Explain that the information comes from external sources.
- State that company IT should validate instructions before
  changing company-managed systems.
- Do not claim external instructions are company policy.
- Do not invent evidence or citations.

Question:
{state["question"]}

External web evidence:
{state["web_results"]}
"""

    response = llm().invoke([HumanMessage(content=prompt)])

    return {
        "answer": response.content.strip(),
        "source_used": "web_search",
        "trace": add_trace(
            state,
            "Answer generation → WEB SEARCH",
        ),
    }

def direct_answer(state: AgentState):
    response = llm().invoke(
        [
            HumanMessage(
                content=(
                    "Respond briefly and naturally to this "
                    f"message: {state['question']}"
                )
            )
        ]
    )

    return {
        "answer": response.content.strip(),
        "source_used": "direct",
        "trace": add_trace(
            state,
            "Direct response → no retrieval",
        ),
    }

def insufficient(state: AgentState):
    return {
        "answer": (
            "I couldn't find enough reliable evidence in the "
            "company knowledge base or external search to answer "
            "confidently. Please contact the IT help desk or "
            "provide more details."
        ),
        "source_used": "insufficient_evidence",
        "trace": add_trace(
            state,
            "Stopped → insufficient reliable evidence",
        ),
    }

def build_graph():
    graph = StateGraph(AgentState)

    nodes = {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }

    for name, function in nodes.items():
        graph.add_node(name, function)

    graph.add_edge(START, "route_question")

    graph.add_conditional_edges(
        "route_question",
        route_after_router,
        {
            "retrieve_kb": "retrieve_kb",
            "direct_answer": "direct_answer",
        },
    )

    graph.add_edge("retrieve_kb", "grade_kb")

    graph.add_conditional_edges(
        "grade_kb",
        after_kb,
        {
            "generate_from_kb": "generate_from_kb",
            "search_web": "search_web",
        },
    )

    graph.add_edge("search_web", "grade_web")

    graph.add_conditional_edges(
        "grade_web",
        after_web,
        {
            "generate_from_web": "generate_from_web",
            "rewrite_query": "rewrite_query",
            "insufficient": "insufficient",
        },
    )

    graph.add_edge("rewrite_query", "retrieve_kb")

    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)

    return graph.compile()

agent_graph = build_graph()

def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }

    return agent_graph.invoke(initial)

