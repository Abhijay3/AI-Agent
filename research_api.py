from fastapi import FastAPI
from pydantic import BaseModel, Field

from research_agent import run_research

app = FastAPI(title="AI Research Assistant")


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=300)


class ResearchResponse(BaseModel):
    report: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    report = run_research(request.topic)
    return ResearchResponse(report=report)
