from langchain_core.messages import HumanMessage, AIMessage
from llm.agent import agent_executor
from uuid import uuid4
import textwrap


def make_user_friendly(text: str) -> str:

    formatted = textwrap.dedent(text).strip()
    return formatted.replace("**", "").replace("\\n", "\n")


def main():
    print("🧠 Welcome to your AI product assistant!\n")
    thread_id = str(uuid4())

    while True:
        user_input = input("💬 Enter your question (or type 'exit' to quit): ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        result = agent_executor.invoke(
            {
                "messages": [HumanMessage(content=user_input)]
            },
            config={"configurable": {"thread_id": thread_id}}
        )

        messages = result.get("messages", [])
        ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]

        if ai_messages:
            raw_response = ai_messages[-1]
            formatted_response = make_user_friendly(raw_response)

            print("\n🤖 Agent response:\n")
            print(formatted_response)
        else:
            print("\n⚠️ No response from agent.")

        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
