from llm.client import get_model
from llm.state_types import State
from typing_extensions import Annotated
from typing import TypedDict


llm = get_model()

class GeneralAnswerOutput(TypedDict):
    """Final response to a general question."""
    answer: Annotated[str, ..., "Direct answer to the user's general knowledge question."]

def create_general_qa_node():
    async def general_qa(state: State) -> State:
        """
        Agent that answers general knowledge or factual questions directly.

        Parameters:
        - state["input_query"]: a natural-language question (e.g., "Who is the president of Italy?")

        Returns:
        - state["answer"]: a concise and accurate response to the question
        """

        prompt = f"""You are a helpful assistant. Answer the following general knowledge question clearly and concisely.

Question: {state['input_query']}
Answer:"""

        
        structured_llm = llm.with_structured_output(GeneralAnswerOutput)
        result = await structured_llm.ainvoke(prompt)

        state["answer"] = result["answer"]
        return state

    return general_qa