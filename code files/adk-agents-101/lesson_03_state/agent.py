from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from lesson_03_state import tools

MODEL = LiteLlm(model="openai/gpt-4o-mini")

INSTRUCTION = """\
You are a travel assistant. Help the user plan short trips.

What you know so far (an empty value means unknown):
- Trip destination: {trip_destination?}
- Number of nights: {trip_nights?}
- Number of travelers: {trip_travelers?}
- Preferred travel style of the user: {user:travel_style?}

Rules:
1. When the user has stated the destination, the nights and the travelers, call save_trip_details.
2. When the user states a travel style (budget, mid-range or luxury), call save_travel_style.
3. Use get_top_attractions to recommend sights, and adapt your tips to the travel style.
4. Use estimate_trip_cost for every cost figure, and never calculate costs yourself.
5. If a tool returns an error, explain the problem to the user.
Answer in at most 150 words.
"""


root_agent = Agent(
    name="my_travel_assistant",
    model=MODEL,
    description="Answers questions about planning short trips.",
    instruction=INSTRUCTION,
    # NEW in Lesson 3: the two state tools join the two tools of Lesson 2.
    tools=[tools.get_top_attractions, tools.estimate_trip_cost,
           tools.save_trip_details, tools.save_travel_style],
)
