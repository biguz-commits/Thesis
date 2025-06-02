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


def display_conversation_history(history):
    """Displays the conversation history in a user-friendly format"""
    if not history:
        print("📝 No previous history")
        return
    
    print("📝 Conversation history:")
    for i, entry in enumerate(history[-6:], 1):  # Show only the last 6 interactions
        role_icon = "👤" if entry["role"] == "user" else "🤖"
        role_name = "You" if entry["role"] == "user" else "Assistant"
        content = entry["content"][:100] + "..." if len(entry["content"]) > 100 else entry["content"]
        print(f"  {role_icon} {role_name}: {content}")
    
    if len(history) > 6:
        print(f"  ... (and {len(history) - 6} more previous interactions)")
    print()


async def run_graph_interactively():
    from llm.graph.build import create_memory_aware_graph  # Assume you renamed the function

    print("🧠 Welcome to your AI product assistant with Memory!\n")
    print("💡 Tip: I can now remember our previous conversations!")
    print("💡 Try questions like: 'Recommend me a laptop' followed by 'Something cheaper'\n")

    memory = InMemorySaver()
    graph = create_memory_aware_graph(checkpointer=memory)
    thread_id = str(uuid4())
    
    print(f"🔗 Session ID: {thread_id[:8]}...\n")

    # Initialize history for this session
    conversation_history = []

    while True:
        user_input = input("💬 Enter your question (or type 'exit' to quit, 'history' to see conversation): ").strip()
        
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == "history":
            display_conversation_history(conversation_history)
            continue
        
        if not user_input:
            print("⚠️ Please enter a valid question.")
            continue

        initial_state = {
            "input_query": user_input,
            "conversation_history": conversation_history.copy(),  # Pass current history
            "session_id": thread_id,
            "answer": "",
            "label": None,
            "context_summary": None
        }

        try:
            print("🔄 Processing your request...")
            final_state = await graph.ainvoke(
                initial_state, 
                config={"configurable": {"thread_id": thread_id}}
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()  # Debug: show full stack trace
            continue

        # Update local history with new messages
        if "conversation_history" in final_state and final_state["conversation_history"]:
            conversation_history = final_state["conversation_history"]

        print("\n📌 Final state of the graph:\n")
        for k, v in final_state.items():
            if k == "conversation_history":
                print(f"{k}: {len(v) if v else 0} messages in history")
            elif k == "input_query":
                # Show if query was enriched
                original_query = user_input
                processed_query = v
                if original_query != processed_query:
                    print(f"input_query (original): {original_query}")
                    print(f"input_query (enhanced): {processed_query}")
                else:
                    print(f"{k}: {v}")
            else:
                print(f"{k}: {v}")

        if "answer" in final_state and final_state["answer"]:
            print("\n🤖 Agent response:\n")
            print(make_user_friendly(final_state["answer"]))
        else:
            print("\n⚠️ No answer produced by the agent.")

        # Show memory stats
        print(f"\n📊 Memory stats: {len(conversation_history)} total messages")
        
        print("\n" + "=" * 80 + "\n")


async def test_memory_scenarios():
    """Test function for specific memory scenarios"""
    from llm.graph.build import create_memory_aware_graph

    print("🧪 Testing Memory Scenarios...\n")
    
    memory = InMemorySaver()
    graph = create_memory_aware_graph(checkpointer=memory)
    thread_id = str(uuid4())
    
    test_scenarios = [
        "Recommend me a laptop for programming",
        "Something cheaper than what you suggested",
        "And for gaming?",
        "I prefer the first one you suggested",
    ]
    
    conversation_history = []
    
    for i, query in enumerate(test_scenarios, 1):
        print(f"🔍 Test {i}: {query}")
        
        initial_state = {
            "input_query": query,
            "conversation_history": conversation_history.copy(),
            "session_id": thread_id,
            "answer": "",
            "label": None,
            "context_summary": None
        }
        
        try:
            result = await graph.ainvoke(
                initial_state, 
                config={"configurable": {"thread_id": thread_id}}
            )
            
            # Update history
            if "conversation_history" in result:
                conversation_history = result["conversation_history"]
            
            print(f"✅ Response: {result.get('answer', 'No answer')[:100]}...")
            print(f"📝 History length: {len(conversation_history)} messages")
            
        except Exception as e:
            print(f"❌ Error in test {i}: {e}")
        
        print("-" * 50)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(test_memory_scenarios())
    else:
        asyncio.run(run_graph_interactively())
