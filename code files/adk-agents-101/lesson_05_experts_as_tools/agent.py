from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext
from lesson_05_experts_as_tools import tools

MODEL = LiteLlm(model="openai/gpt-4o-mini")

destination_expert = Agent(
    name="destination_expert",
    model=MODEL,
    # The root agent receives this description as the description of a tool.
    description="Recommends sights and activities for the trip destination.",
    instruction="""\
You are the destination expert of a travel team.
You receive one request from the travel planner and answer it in one reply.
Trip: {trip_destination?}, {trip_nights?} nights, {trip_travelers?} travelers.
Travel style of the user: {user:travel_style?}
Recommend sights with get_top_attractions, adapt them to the travel style,
and spread them over the days of the trip. Answer in at most 120 words.
""",
    tools=[tools.get_top_attractions],
    # NEW in Lesson 5: in single-turn mode, ADK offers this agent to its parent
    # as a tool with the name of the agent and one argument, request.
    mode="single_turn",
    # NEW in Lesson 5: ADK saves the final answer of the agent in state under
    # this key, where other agents and code can read it.
    output_key="sights_plan",
    # NEW in Lesson 5: ADK 2.13.0 still offers a single-turn agent the tool
    # transfer_to_agent, with the parent as its target. A transfer would leave
    # the root agent with an empty result, so this flag removes the tool.
    disallow_transfer_to_parent=True,
)

budget_expert = Agent(
    name="budget_expert",
    model=MODEL,
    description="Estimates the cost of the trip.",
    instruction="""\
You are the budget expert of a travel team.
You receive one request from the travel planner and answer it in one reply.
Trip: {trip_destination?}, {trip_nights?} nights, {trip_travelers?} travelers.
Calculate every cost with estimate_trip_cost, and never calculate costs yourself.
If a trip detail is unknown, name the missing detail in your answer.
Answer in at most 80 words.
""",
    tools=[tools.estimate_trip_cost, tools.convert_currency],
    # NEW in Lesson 5: the same three settings as for the destination expert.
    mode="single_turn",
    output_key="cost_estimate",
    disallow_transfer_to_parent=True,
)

ROOT_INSTRUCTION = """\
You are the travel planner, the first contact of the user. You lead a team of two experts,
which you call as tools.

What you know so far (an empty value means unknown):
- Trip destination: {trip_destination?}
- Number of nights: {trip_nights?}
- Number of travelers: {trip_travelers?}
- Preferred travel style of the user: {user:travel_style?}

Rules:
1. Ask for any missing trip detail: the destination, the number of nights and the number of travelers.
2. When all three details are known, call save_trip_details.
3. When the user states a travel style (budget, mid-range or luxury), call save_travel_style.
4. For sights and activities, call destination_expert. For costs, call budget_expert.
5. The experts do not see this conversation, so write every request as a complete task.
6. When the user asks for a plan, call both experts and combine their answers into one plan.
"""

root_agent = Agent(
    name="travel_planner",
    model=MODEL,
    description="Plans trips with a team of two experts.",
    instruction=ROOT_INSTRUCTION,
    tools=[tools.save_trip_details, tools.save_travel_style],
    # The experts stay in the list of sub-agents. Because both experts work in
    # single-turn mode, ADK adds them to the tools of the root agent instead
    # of offering a transfer to them.
    sub_agents=[destination_expert, budget_expert],
)
