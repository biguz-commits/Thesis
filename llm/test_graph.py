# run with python -m llm.test_graph from the project root

import asyncio

from uuid import uuid4
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_graph_sync():
    from llm.graph.build import create_graph  

    graph = create_graph()

    input_query = "Can you reccomend me a tv under 300$ ?"
    initial_state = {"input_query": input_query}

    final_state = asyncio.run(
        graph.ainvoke(
            initial_state
    ))

    print("📌 Final state of the graph:")
    for k, v in final_state.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    test_graph_sync()
