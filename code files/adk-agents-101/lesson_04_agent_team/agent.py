from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext
from lesson_04_agent_team import tools

MODEL = LiteLlm(model="openai/gpt-4o-mini")

destination_expert = Agent(
    name="destination_expert",
    model=MODEL,
    # The other agents read this description when they decide whether to
    # transfer a question to this agent, so the description must be precise.
    description="Recommends sights and activities for the trip destination.",
    instruction="""\
You are the destination expert of a travel team.
Trip: {trip_destination?}, {trip_nights?} nights, {trip_travelers?} travelers.
Travel style of the user: {user:travel_style?}
Recommend sights with get_top_attractions, adapt them to the travel style,
and spread them over the days of the trip. Answer in at most 150 words.
For questions about costs, transfer to budget_expert.
For any other question, transfer to travel_planner.
""",
    tools=[tools.get_top_attractions],
)


budget_expert = Agent(
    name="budget_expert",
    model=MODEL,
    description="Estimates the cost of the trip.",
    instruction="""\
You are the budget expert of a travel team.
Trip: {trip_destination?}, {trip_nights?} nights, {trip_travelers?} travelers.
Calculate every cost with estimate_trip_cost, and never calculate costs yourself.
Answer in at most 100 words.
If a trip detail is unknown, transfer to travel_planner, which collects the details.
For questions about sights, transfer to destination_expert.
For any other question, transfer to travel_planner.
""",
    tools=[tools.estimate_trip_cost, tools.convert_currency],
)


ROOT_INSTRUCTION = """\
You are the travel planner, the first contact of the user. You lead a team of two experts.

What you know so far (an empty value means unknown):
- Trip destination: {trip_destination?}
- Number of nights: {trip_nights?}
- Number of travelers: {trip_travelers?}
- Preferred travel style of the user: {user:travel_style?}

Rules:
1. Ask for any missing trip detail: the destination, the number of nights and the number of travelers.
2. When all three details are known, call save_trip_details.
3. When the user states a travel style (budget, mid-range or luxury), call save_travel_style.
4. Transfer questions about sights and activities to destination_expert.
5. Transfer questions about costs to budget_expert.
"""

root_agent = Agent(
    # The root agent has a new name that matches its new role: it leads a team.
    name="travel_planner",
    model=MODEL,
    description="Plans trips with a team of two experts.",
    instruction=ROOT_INSTRUCTION,
    # The root agent keeps the two state tools, because it collects the trip details.
    tools=[tools.save_trip_details, tools.save_travel_style],
    # NEW in Lesson 4: the sub-agents of the root agent. Because of this list,
    # ADK gives every agent of the team a tool named transfer_to_agent, with
    # which the model can hand the conversation to another agent of the team.
    sub_agents=[destination_expert, budget_expert],
)
