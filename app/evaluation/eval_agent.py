import os
import sys
import csv
import json
import asyncio
from typing import List, Dict

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from app.llm.client import get_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class InstructionOutput(BaseModel):
    incorrect_count: int = Field(..., description="Number of non-compliant or incorrect answers")
    patterns: List[str] = Field(..., description="Short bullet patterns spotted across incorrect answers; empty if none")


def build_prompt_template() -> ChatPromptTemplate:
    system = (
        "You are an evaluation agent for an Amazon-like product recommendation system. "
        "Your task is to judge each (question, answer) pair strictly from the provided text—no external knowledge, no web browsing.\n\n"
        "EVALUATION CRITERIA:\n"
        "1) Recommendation-type questions (e.g., 'recommend', 'suggest', categories like fashion/electronics/books/beauty/automotive/video games):\n"
        "   - The answer must be actionable for purchase: concrete product advice with real brand/model names (no placeholders like 'brand x', 'headphone 1').\n"
        "   - If the question implies a budget (e.g., 'under $50'), the answer should include a price (exact or range) consistent with the constraint.\n"
        "   - References to reviews/ratings or social proof should be meaningful (avoid fabricated, contextless numbers).\n"
        "   - Respect explicit constraints (e.g., 'not too big', platform, skin type, age range).\n"
        "2) General knowledge questions (e.g., 'Who is Taylor Swift?', 'What is Cristiano Ronaldo famous for?'):\n"
        "   - The answer must be relevant and coherent to the question; off-topic answers are incorrect.\n"
        "\n"
        "SCORING RULE:\n"
        "- Mark a pair as INCORRECT if it violates any core expectation above (not actionable, placeholders, missing price when budget is explicit, irrelevant GK answer, etc.).\n"
        "- Otherwise mark it as CORRECT.\n"
        "\n"
        "STRICT CONSISTENCY REQUIREMENTS:\n"
        "- Report ONLY patterns that were ACTUALLY OBSERVED among the INCORRECT answers.\n"
        "- If incorrect_count = 0, then patterns MUST be an empty list [].\n"
        "- Do NOT include hypothetical, generic, or predictive patterns.\n"
        "- Patterns must be concise (≤ 7 words) and non-duplicative.\n"
        "- Be conservative: if you are unsure, prefer marking CORRECT.\n"
        "\n"
        "OUTPUT FORMAT:\n"
        "- Return a structured object with exactly these fields:\n"
        "  incorrect_count: integer\n"
        "  patterns: list of strings (empty if incorrect_count = 0)\n"
        "No extra fields, no explanations."
    )

    user = (
        "Evaluate the following Q/A pairs. Input is JSONL; each line has fields 'question' and 'answer'.\n\n"
        "JSONL START\n{jsonl}\nJSONL END\n\n"
        "Return ONLY the structured object described above. "
        "Remember: if no answers are incorrect, set incorrect_count = 0 and patterns = []. "
        "Include patterns only if they were truly observed among the incorrect answers."
    )

    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("user", user),
    ])


def load_qa(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {c for c in ["question", "answer"] if c not in (reader.fieldnames or [])}
        if missing:
            raise ValueError(f"CSV must contain columns {missing}. Found: {reader.fieldnames}")
        for r in reader:
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            if q:
                rows.append({"question": q, "answer": a})
    if not rows:
        raise ValueError("No rows found in CSV.")
    return rows

def to_jsonl(rows: List[Dict[str, str]]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)


async def evaluate_csv(csv_path: str, llm):

    qa = load_qa(csv_path)
    jsonl = to_jsonl(qa)

    instruction_prompt_template = build_prompt_template()
    prompt = instruction_prompt_template.invoke({"jsonl": jsonl})

    structured_llm = llm.with_structured_output(InstructionOutput)
    result: InstructionOutput = await structured_llm.ainvoke(prompt)

    return {
        "incorrect_count": result.incorrect_count,
        "patterns": result.patterns or []
    }


async def main(input_csv: str):
    llm = get_model(model_name="meta-llama/llama-3.1-8b-instruct")
    result = await evaluate_csv(input_csv, llm)
    print(result)


if __name__ == "__main__":
    input_csv = "app/evaluation/output/results.csv"
    asyncio.run(main(input_csv))
