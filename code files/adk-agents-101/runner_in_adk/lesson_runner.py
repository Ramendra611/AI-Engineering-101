from google.adk.models.lite_llm import LiteLlm
# from .agent import root_agent
from google.genai import types
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.adk.memory import InMemoryMemoryService
import asyncio
from tools import convert_currency, get_top_attractions, estimate_trip_cost
# from lesson_03_state import tools
from google.adk.tools import load_memory

from dotenv import load_dotenv

# # adk run and adk web load the file .env automatically, but a program of your
# # own must load it itself. The call precedes the ADK imports, because LiteLLM
# # reads the setting LITELLM_LOG at the moment Python imports it.
load_dotenv()


# The model string has the form "provider/model". Here the provider is OpenAI
# and the model is GPT-6 Luna. LiteLLM reads the API key from the environment
# variable OPENAI_API_KEY, which ADK loads from the file .env.
MODEL = LiteLlm(model="openai/gpt-4o-mini")


INSTRUCTION = """\
You are a travel assistant. Help the user plan short trips.
Use the tool get_top_attractions to recommend sights.
Use the tool estimate_trip_cost for every cost figure, and never calculate costs yourself.
Use the tool convert_currency whenever the user asks to convert prices or budgets between currencies.
If a tool returns an error, explain the problem to the user.

If the user asks about information from previous
conversations, use the load_memory tool to retrieve
relevant information before answering.
Answer in at most 150 words.
"""


# The agent is a configuration object. Creating the object does not call the
# model; ADK calls the model only when the user sends a message.
root_agent = Agent(
    # A unique name. ADK shows this name as the author of the agent's replies.
    # The model that generates the replies of the agent.
    name="travel_assistant",
    model=MODEL,
    description="Answers questions about planning short trips.",
    # The instruction defines the role and the rules of the agent. ADK sends
    # this text to the model at the start of every request.
    instruction=INSTRUCTION,
    tools=[convert_currency, get_top_attractions,
           estimate_trip_cost],
)


# The name of the app and the identifier of the user. The services file every
# session and every memory under this pair of values.
APP_NAME = "travel_planner"
USER_ID = "traveler_1"


def print_event(event) -> None:
    """Print one event of the event stream as one or more short lines.

    Args:
        event: An event that the runner has produced.
    """
    # print("event:  ***", event)
    for call in event.get_function_calls():
        print(f"[{event.author}] calls {call.name} with {call.args}")
    for response in event.get_function_responses():
        print(f"[{event.author}] receives the result of {response.name}")
    # A final response is the last event of an agent's turn: its answer.
    if event.is_final_response() and event.content and event.content.parts:
        text = "".join(part.text or "" for part in event.content.parts)
        print(f"[{event.author}] answers: {text}")


async def send(runner: Runner, session_id: str, text: str) -> None:
    """Send one message of the user to the agent and print the events of the invocation.

    Args:
        runner: The runner that executes the agent.
        session_id: The identifier of the session that receives the message.
        text: The message of the user.
    """
    print(f"\n[user] {text}")
    # ADK represents every message as Content: a role and a list of parts.
    message = types.Content(role="user", parts=[types.Part(text=text)])
    # run_async returns an asynchronous stream of events. The loop receives
    # each event as soon as ADK has created it.
    async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=message):
        print_event(event)


async def main() -> None:
    """Create the services and the runner, and hold two conversations."""
    # The session service stores sessions with their events and state. The
    # memory service stores past conversations for later searches. Both keep
    # their data in the memory of this process, so the data ends with it.
    session_service = InMemorySessionService()
    memory_service = InMemoryMemoryService()

    # The runner executes the agent for each message and connects it to the services.
    runner = Runner(
        app_name=APP_NAME,
        agent=root_agent,
        session_service=session_service,
        memory_service=memory_service,
    )

    # The first conversation plans a trip.
    first = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    await send(runner, first.id, "Plan 2 nights in Kyoto for one person. We like budget travel.")

    completed_session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=first.id
    )

    await memory_service.add_session_to_memory(
        completed_session
    )

    # get_session returns the stored session, including the state that the tools wrote.
    stored = await session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=first.id)
    print("\nState of the first session:", stored.state)

    # The second conversation starts without events, but the memory service
    # already holds the first conversation, because the callback saved it.
    second = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    await send(runner, completed_session.id, "Which city did I plan last time, and what did it cost?")


if __name__ == "__main__":
    # asyncio.run starts an event loop, runs main to its end and closes the loop.
    asyncio.run(main())
