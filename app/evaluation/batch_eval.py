# run with: python -m app.evaluation.batch_eval --in questions.csv --out answers.csv --col question


import os
import sys
import csv
import argparse
import asyncio
from uuid import uuid4
import textwrap
from typing import List, Tuple, Optional


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def make_user_friendly(text: str) -> str:
    formatted = textwrap.dedent(text).strip()
    return formatted.replace("**", "").replace("\\n", "\n")


def read_questions(csv_path: str, column: str, limit: int = 50) -> List[str]:
    questions: List[str] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if column not in reader.fieldnames:
            raise ValueError(
                f"La colonna '{column}' non è presente nel CSV. Colonne disponibili: {reader.fieldnames}"
            )
        for row in reader:
            q = (row.get(column) or "").strip()
            if q:
                questions.append(q)
            if len(questions) >= limit:
                break
    if not questions:
        raise ValueError("Nessuna domanda valida trovata nel file di input.")
    return questions


async def ask_one(graph, question: str) -> Tuple[str, str]:
    """Invoca il grafo per una singola domanda, restituisce (question, answer)."""
    thread_id = str(uuid4())
    initial_state = {
        "input_query": question,
        "thread_id": thread_id,
    }
    try:
        final_state = await graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )
        answer: Optional[str] = None
        if isinstance(final_state, dict):
            answer = final_state.get("answer")
            if answer is None:
                answer = str({k: (v if isinstance(v, (str, int, float)) else str(v)) for k, v in final_state.items()})
        else:
            answer = str(final_state)
        return question, make_user_friendly(answer or "")
    except Exception as e:
        return question, f"ERROR: {e}"


async def run_batch(input_csv: str, output_csv: str, column: str, concurrency: int = 3):
    from langgraph.checkpoint.memory import InMemorySaver
    from app.llm.graph.build import create_graph

    memory = InMemorySaver()
    graph = create_graph(checkpointer=memory)

    questions = read_questions(input_csv, column, limit=50)

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _guard(q: str) -> Tuple[str, str]:
        async with sem:
            return await ask_one(graph, q)

    tasks = [asyncio.create_task(_guard(q)) for q in questions]
    results: List[Tuple[str, str]] = await asyncio.gather(*tasks)


    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer"])
        for q, a in results:
            writer.writerow([q, a])


def main():
    parser = argparse.ArgumentParser(description="Batch Q&A verso il grafo LangGraph.")
    parser.add_argument("--in", dest="input_csv", required=True, help="Path al CSV di input con le domande.")
    parser.add_argument("--out", dest="output_csv", required=True, help="Path del CSV di output (domande/risposte).")
    parser.add_argument(
        "--col",
        dest="column",
        default="question",
        help="Nome della colonna nel CSV di input che contiene le domande (default: 'question').",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Numero massimo di richieste in parallelo (default: 3). Imposta a 1 per esecuzione sequenziale.",
    )
    args = parser.parse_args()

    asyncio.run(run_batch(args.input_csv, args.output_csv, args.column, args.concurrency))


if __name__ == "__main__":
    main()
