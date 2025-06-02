from llm.client import get_model
from llm.state_types import State
from typing_extensions import Annotated
from typing import TypedDict
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

llm = get_model()

class GeneralAnswerOutput(BaseModel):
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

        parser = PydanticOutputParser(pydantic_object=GeneralAnswerOutput)

        prompt_template = PromptTemplate(
            template="""You are a helpful assistant. Answer the following general knowledge question clearly and concisely.

Question: {question}

{format_instructions}

Answer:""",
            input_variables=["question"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )


        chain = prompt_template | llm | parser
        
        try:
            result = await chain.ainvoke({"question": state['input_query']})
            state["answer"] = result.answer
            
        except Exception as e:
            logger.warning(f"Structured output failed: {e}")

            try:
                fallback_prompt = f"""You are a helpful assistant. Answer the following general knowledge question clearly and concisely.

Question: {state['input_query']}
Answer:"""
                
                raw_result = await llm.ainvoke(fallback_prompt)

                if hasattr(raw_result, 'content'):
                    state["answer"] = raw_result.content.strip()
                else:
                    state["answer"] = str(raw_result).strip()
                    
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                state["answer"] = "Sorry but I cannot process your question at the moment"

        return state

    return general_qa