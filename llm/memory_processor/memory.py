from llm.client import get_model
from llm.state_types import State
from typing import List, Dict

def create_memory_processor_node():
    llm = get_model()
    
    async def memory_processor(state: State) -> State:
        """
        Preprocess the query by considering the conversational context
        """

        if "conversation_history" not in state or state["conversation_history"] is None:
            state["conversation_history"] = []
        
        current_query = state["input_query"]
        history = state["conversation_history"]

        if history:
            context_prompt = f"""
Based on the previous conversation history, reformulate or enrich the following query to make it clearer and more specific.

Conversation history:
{format_conversation_history(history)}

Current query: {current_query}

If the current query refers to elements from the previous conversation (such as "that one", "that product", "as before", etc.), 
include the necessary details to make it self-contained.

Enriched query:"""

            try:
                enriched_response = await llm.ainvoke(context_prompt)
                enriched_query = enriched_response.content.strip() if hasattr(enriched_response, 'content') else str(enriched_response).strip()

                if len(enriched_query) > len(current_query) * 1.2:
                    state["input_query"] = enriched_query
                    
            except Exception as e:
                print(f"Error during query enrichment: {e}")

        return state
    
    return memory_processor

def format_conversation_history(history: List[Dict[str, str]], max_entries: int = 10) -> str:
    """Format the conversation history for use in a prompt"""
    recent_history = history[-max_entries:] if len(history) > max_entries else history
    
    formatted = []
    for entry in recent_history:
        role = "User" if entry["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {entry['content']}")
    
    return "\n".join(formatted)
