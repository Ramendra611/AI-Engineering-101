from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from .tools import convert_currency, get_top_attractions, estimate_trip_cost

MODEL = LiteLlm(model="openai/gpt-4o-mini")


INSTRUCTION = """\
You are a travel assistant. Help the user plan short trips.
Use the tool get_top_attractions to recommend sights.
Use the tool estimate_trip_cost for every cost figure, and never calculate costs yourself.
Use the tool convert_currency whenever the user asks to convert prices or budgets between currencies.
If a tool returns an error, explain the problem to the user.
Answer in at most 150 words.
"""

root_agent = Agent(
    # A unique name. ADK shows this name as the author of the agent's replies.
    # The model that generates the replies of the agent.
    name="travel_assistant",
    model=MODEL,
    description="Answers questions about planning short trips.",
    # The instruction defines the role and the rules of the agent. ADK sends
    # this text to the model at the start of every request.
    instruction=INSTRUCTION,
    tools=[convert_currency, get_top_attractions, estimate_trip_cost],
)
