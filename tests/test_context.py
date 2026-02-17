import asyncio

from app.services.context_manager import ContextManager


def test_manual():
    cm = ContextManager()
    conv_id = "test_123"

    print("Step 1: Storing memories...")
    cm.store_memory(conv_id, "The capital of France is Paris.")
    cm.store_memory(conv_id, "The project name is Horizon.")

    print("Step 2: Testing semantic search...")
    # We ask about "France" but expect the full sentence back
    result = cm.search_context(conv_id, "What is the capital?")
    print(f"Search Result:\n{result}")

    if "Paris" in result:
        print("✅ Semantic Search Passed!")
    else:
        print("❌ Semantic Search Failed.")

    print("\nStep 3: Checking SQLite storage...")
    messages = cm.get_conversation_messages(conv_id)
    print(f"Found {len(messages)} messages in SQLite.")


if __name__ == "__main__":
    test_manual()
