from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from llm.tools.db_tool import query_purchases
from llm.tools.vectordb_tool import query_reviews
from llm.client import get_model


system_prompt = """You are a helpful product recommendation assistant. 
Use query_reviews to search for similar product reviews based on user descriptions. 
Use query_purchases to fetch structured product details such as price and rating. 
Always justify your suggestions with review evidence and product stats."""

model = get_model()

memory = MemorySaver()


tools = [query_reviews, query_purchases]

agent_executor = create_react_agent(
    model=model,
    tools=tools,
    checkpointer=memory,
    prompt=system_prompt
)