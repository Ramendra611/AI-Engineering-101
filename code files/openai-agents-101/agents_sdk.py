"""The travel agent rebuilt with the OpenAI Agents SDK.

The SDK supplies what wa earlier built by hand: the loop (Runner), the tool
schemas (function_tool reads type hints and docstrings) and the session
(SQLiteSession). What remains is a few declarations.
"""

import os
from config import LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
from agents import (Agent, OpenAIChatCompletionsModel, Runner, SQLiteSession,
                    function_tool, set_tracing_disabled)
from openai import AsyncOpenAI

import travel_tools as tt


def pick_model():
    """OpenAI by default. If LLM_BASE_URL is set (for example Ollama), use that server instead."""
    if os.getenv("LLM_BASE_URL"):
        # traces are uploaded to OpenAI, which needs an OpenAI key
        # set_tracing_disabled(True)
        return OpenAIChatCompletionsModel(
            model=os.getenv("LLM_MODEL", "qwen3:4b"),
            openai_client=AsyncOpenAI(base_url=os.environ["LLM_BASE_URL"],
                                      api_key=os.getenv("LLM_API_KEY", "ollama")))
    return os.getenv("LLM_MODEL", "gpt-4o-mini")


travel_agent = Agent(
    name="Travel assistant",
    instructions="You help plan trips. Use the tools for weather, flights and currency. Be brief.",
    tools=[function_tool(tt.get_weather), function_tool(tt.search_flights),
           function_tool(tt.convert_currency)],
    model=pick_model(),
)


print("Configured model:", LLM_MODEL)
print("Configured endpoint:", LLM_BASE_URL)
print("Provider:", "Ollama" if LLM_BASE_URL else "OpenAI")

if __name__ == "__main__":
    # history is stored in SQLite
    session = SQLiteSession("suresh-goa-trip", "conversations.db")

    result = Runner.run_sync(travel_agent, "What's the cheapest flight from goa to maharashtra?",
                             session=session)
    print("assistant:", result.final_output)
    for item in result.new_items:  # everything that happened inside this run
        print("   ", item.type)
        print('\n\n')

    result = Runner.run_sync(
        travel_agent, "And how's the weather there? what should i pack? should i pack warm clothes?", session=session)
    # "there" is resolved from the stored history
    print("assistant:", result.final_output)
