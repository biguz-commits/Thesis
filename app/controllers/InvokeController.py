import textwrap
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from app.llm.graph.build import create_graph


class InvokeController:
    def __init__(self):
        self.memory = InMemorySaver()

    @staticmethod
    def make_user_friendly(text: str) -> str:
        formatted = textwrap.dedent(text).strip()
        return formatted.replace("**", "").replace("\\n", "\n")


    async def invoke(self, user_input: str):
        graph = create_graph(checkpointer=self.memory)
        thread_id = str(uuid4())
        initial_state = {
            "input_query": user_input,
            "thread_id": thread_id,
        }
        final_state = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": thread_id}})

        if "answer" in final_state:
            response = self.make_user_friendly(final_state["answer"])
            return response





