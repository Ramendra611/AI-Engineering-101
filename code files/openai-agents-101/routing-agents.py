import asyncio

from agents import Agent, Runner
from config import OPENAI_API_KEY
import pprint


history_tutor = Agent(
    name="History tutor",
    handoff_description="Specialist for history questions.",
    instructions="Answer history questions clearly and concisely.",
    model="gpt-4o-mini"
)

math_tutor = Agent(
    name="Math tutor",
    handoff_description="Specialist for math questions.",
    instructions="Explain math step by step and include worked examples.",
    model="gpt-4o-mini"
)


triage_agent = Agent(
    name="Homework triage",
    instructions="Route each homework question to the right specialist.",
    handoffs=[history_tutor, math_tutor],
    model="gpt-4o-mini"
)


async def main() -> None:
    result = await Runner.run(
        triage_agent,
        "Who was the first president of the United States?",
    )
    print(result.final_output)
    print(result.last_agent.name)
    print("\n\n")

    # for item in result.new_items:
    #     print("=" * 80)
    #     print(type(item).__name__)

    #     # Try the most common structured representations
    #     if hasattr(item, "model_dump"):
    #         pprint.pp(item.model_dump())
    #     elif hasattr(item, "dict"):
    #         pprint.pp(item.dict())
    #     else:
    #         pprint.pp(vars(item))
    #     print("=" * 80)



if __name__ == "__main__":
    asyncio.run(main())
