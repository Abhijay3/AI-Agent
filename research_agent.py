import os

from dotenv import load_dotenv
from openai import OpenAI

from tools import web_search

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
)

MODEL = "llama-3.3-70b-versatile"


def ask(system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def planner_agent(topic: str) -> list:
    text = ask(
        "You are a research planning agent. Given a topic, output exactly 3 "
        "specific sub-questions that would help research it thoroughly. "
        "Output ONLY the questions, one per line, no numbering, no extra text.",
        topic,
    )
    return [q.strip() for q in text.splitlines() if q.strip()]


def research_sub_question(question: str) -> str:
    results = web_search(question, max_results=3)
    return ask(
        "You are a research agent. Based on the search results below, write "
        "a concise, factual 2-3 sentence answer to the question. If the "
        "results don't answer it, say so honestly instead of guessing.",
        f"Question: {question}\n\nSearch results:\n{results}",
    )


def writer_agent(topic: str, findings: list) -> str:
    findings_text = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in findings)
    return ask(
        "You are a research report writer. Synthesize the following research "
        "findings into a well-structured markdown report: a title, a short "
        "summary paragraph, then one section per finding with a heading.",
        f"Topic: {topic}\n\nFindings:\n{findings_text}",
    )


def run_research(topic: str) -> str:
    questions = planner_agent(topic)
    findings = [(q, research_sub_question(q)) for q in questions]
    return writer_agent(topic, findings)


if __name__ == "__main__":
    topic = input("Research topic: ")
    print(run_research(topic))
