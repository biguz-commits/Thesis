from langgraph.graph import StateGraph
from llm.state_types import State  

from langgraph.checkpoint.memory import MemorySaver  

from llm.graph.nodes.bert_node import create_bert_classifier_node
from llm.graph.nodes.general_agent import create_general_qa_node
from llm.graph.nodes.reccomandation_agent import create_recommendation_node

def router(state: State) -> str:
    label = state["label"]
    return "recommendation" if label == 1 else "general_qa"

def create_graph(checkpointer : MemorySaver) -> StateGraph:
    builder = StateGraph(State)

    builder.add_node("bert_classifier", create_bert_classifier_node())
    builder.add_node("recommendation", create_recommendation_node())
    builder.add_node("general_qa", create_general_qa_node())

    builder.add_conditional_edges("bert_classifier", router)

    builder.set_entry_point("bert_classifier")
    builder.set_finish_point("recommendation")
    builder.set_finish_point("general_qa")

    
    return builder.compile(checkpointer=checkpointer)
