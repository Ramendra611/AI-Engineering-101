"""Lesson 1: the smallest useful ADK agent.

This file defines one agent that answers questions about planning a trip.
ADK's command-line tools (adk run and adk web) import this file and look for
a variable named root_agent, so that variable name must stay exactly as it is.
"""

# Agent is ADK's class for an agent whose decisions come from a language model.
# Its full name is LlmAgent; Agent is a shorter name for the same class.
from google.adk.agents import Agent

# LiteLlm is ADK's adapter for models that are not Gemini models. The adapter
# sends each request to the model provider through the LiteLLM library.
from google.adk.models.lite_llm import LiteLlm

# The model string has the form "provider/model". Here the provider is OpenAI
# and the model is GPT-6 Luna. LiteLLM reads the API key from the environment
# variable OPENAI_API_KEY, which ADK loads from the file .env.
MODEL = LiteLlm(model="openai/gpt-4o-mini")

# The agent is a configuration object. Creating the object does not call the
# model; ADK calls the model only when the user sends a message.
root_agent = Agent(
    # A unique name. ADK shows this name as the author of the agent's replies.
    name="travel_assistant",# The model that generates the replies of the agent.
    model=MODEL,
    description="Answers questions about planning short trips.",
    # The instruction defines the role and the rules of the agent. ADK sends
    # this text to the model at the start of every request.
    instruction=(
        "You are a travel assistant. "
        "Help the user plan short trips: suggest destinations, sights and practical tips. "
        "Answer in at most 120 words."
    ),
)
