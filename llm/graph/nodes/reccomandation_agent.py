from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage
from typing import TypedDict
from typing_extensions import Annotated

from llm.state_types import State
from llm.client import get_model
from llm.tools.db_tool import query_purchases
from llm.tools.vectordb_tool import query_reviews
from langchain_core.prompts import ChatPromptTemplate


def get_recommendation_prompt() -> ChatPromptTemplate:
    system_prompt = """
You are a helpful product recommendation assistant.

You MUST always call at least one tool in your response.
NEVER answer directly without calling a tool.

Your job is to suggest the best products based on both structured data (from the purchases database) and unstructured data (from user reviews).

You have access to the following tools:

---

 `query_reviews`  
Searches for the most similar reviews to a user query in a large vector database of product reviews.

- **Input**:
    - `query_text`: natural language description of the product or requirement
    - `top_k` (optional): number of similar reviews to return (default: 8)

- **Output**:
    - A JSON list of the most similar reviews, each including:
        - The original review text
        - Metadata such as product or reviewer info
        - A similarity score

Use this tool when the user mentions preferences, product types, or descriptive feedback.

---

`query_purchases`  
Retrieves product and transaction information from a relational database by joining the `purchases` and `products` tables.

- **Input filters**:
    - `start_date`: only purchases from this date forward (YYYY-MM-DD)
    - `end_date`: only purchases up to this date
    - `max_price`: only products with price below or equal to this value
    - `title_contains`: keyword in the product title

- **Output**:
    - For each matching product:
        - Price paid
        - Product title
        - Average rating
        - Number of ratings

Use this tool when the user mentions pricing, product categories, or wants to compare options.

---

INSTRUCTIONS:

1. You MUST always call at least one tool. If the user query is vague, call both tools with empty or default parameters.
2. NEVER return raw tool outputs. After gathering the data, rephrase and summarize it into a clear, structured, **human-friendly recommendation**.
3. Present your answer as a coherent, readable paragraph or bullet points, not JSON or raw data.
4. Do NOT answer general knowledge or unrelated questions.
5. Do NOT generate fictional data. Base your answer only on tool results.
6. If no data is returned by the tools, inform the user politely that no products were found matching the criteria.



""".strip()

    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input_query}"),
    ])


def create_recommendation_node():
    llm: BaseChatModel = get_model()

    tools: list[BaseTool] = [query_reviews, query_purchases]

    prompt = get_recommendation_prompt()
    llm_with_tools = llm.bind_tools(tools)
    runnable = prompt | llm_with_tools

    async def recommend(state: State) -> State:
        query = state["input_query"]
        if not query:
            raise ValueError("Missing 'input_query' in state")

        tools_by_name = {tool.name: tool for tool in tools}

        
        ai_message: BaseMessage = await runnable.ainvoke({"input_query": query})


        if not hasattr(ai_message, "tool_calls") or not ai_message.tool_calls:
          state["answer"] = ai_message.content
          return state


        results = {}
        
        
        for call in ai_message.tool_calls:
            tool_name = call["name"]
            args = call["args"]
            tool = tools_by_name.get(tool_name)

            if not tool:
              results[tool_name] = f"[ Tool '{tool_name}' not found]"
              continue

            print(f" Calling tool `{tool_name}` with args: {args}")
            result = await tool.ainvoke(args)
            results[tool_name] = result

        
        state["answer"] = str(results)
        return state

    return recommend
