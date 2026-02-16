import asyncio

from llama_cpp import CreateChatCompletionStreamResponse, Iterator

from app.services.orchestrator import Orchestrator


async def main():
    # Initialize Orchestrator
    orchestrator = Orchestrator()

    # final_response: Iterator[
    #     CreateChatCompletionStreamResponse
    # ] = await orchestrator.execute_orchestration(
    #     user_prompt="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    #     conversation_id="28654632368",
    # )

    # Assuming 'final_response' is your Iterator[CreateChatCompletionStreamResponse]
    print("\n--- JUDGE SYNTHESIS START ---\n")

    # for chunk in final_response:
    #     # 1. Extract the 'delta' dictionary from the first choice
    #     delta = chunk["choices"][0]["delta"]

    #     # 2. Check if 'content' exists (first and last chunks might be empty or metadata)
    #     if "content" in delta:
    #         content = delta.get("content")

    #         # 3. Print to terminal without a newline to simulate 'typing'
    #         print(content, end="", flush=True)

    print("\n\n--- JUDGE SYNTHESIS END ---")


if __name__ == "__main__":
    asyncio.run(main())
