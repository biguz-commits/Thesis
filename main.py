# activate the venv with source $(poetry env info --path)/bin/activate
# run with: python -m src.thesis.main

import asyncio
from uuid import uuid4
import sys
import textwrap
from langgraph.checkpoint.memory import InMemorySaver

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def make_user_friendly(text: str) -> str:
    formatted = textwrap.dedent(text).strip()
    return formatted.replace("**", "").replace("\\n", "\n")


async def run_graph_interactively():
    from app.llm.graph.build import create_graph

    print(" Welcome to your AI product assistant!\n")

    memory = InMemorySaver()

    graph = create_graph(checkpointer=memory)
    thread_id = str(uuid4())

    while True:
        user_input = input("Enter your question (or type 'exit' to quit): ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        initial_state = {
            "input_query": user_input,
            "thread_id": thread_id,
        }

        try:
            final_state = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})
        except Exception as e:
            print(f" Error: {e}")
            continue

        print("\n Final state of the graph:\n")
        for k, v in final_state.items():
            print(f"{k}: {v}")

        if "answer" in final_state:
            print("\n Agent response:\n")
            print(make_user_friendly(final_state["answer"]))
        else:
            print("\n No answer produced by the agent.")

        print("\n" + "=" * 80 + "\n")


def run_dev():
    asyncio.run(run_graph_interactively())

if __name__ == "__main__":
    asyncio.run(run_graph_interactively())
