# TODO: 1 - Import the KnowledgeAugmentedPromptAgent and RoutingAgent
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

# TODO: 2 - Define the Texas Knowledge Augmented Prompt Agent
texas_persona = "You are a college professor"
texas_knowledge = "You know everything about Texas"
texas_agent = KnowledgeAugmentedPromptAgent(openai_api_key, texas_persona, texas_knowledge)

# TODO: 3 - Define the Europe Knowledge Augmented Prompt Agent
europe_knowledge = "You know everything about Europe"
europe_agent = KnowledgeAugmentedPromptAgent(openai_api_key, texas_persona, europe_knowledge)

# TODO: 4 - Define the Math Knowledge Augmented Prompt Agent
math_persona = "You are a college math professor"
math_knowledge = "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
math_agent = KnowledgeAugmentedPromptAgent(openai_api_key, math_persona, math_knowledge)

routing_agent = RoutingAgent(openai_api_key, {})
agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        "func": lambda x: texas_agent.respond(x) # TODO: 5 - Call Texas Agent
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": lambda x: europe_agent.respond(x) # TODO: 6 - Define Europe function
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        "func": lambda x: math_agent.respond(x) # TODO: 7 - Define Math function
    }
]

routing_agent.agents = agents

# TODO: 8 - Print the RoutingAgent responses
prompts = [
    "Tell me about the history of Rome, Texas",
    "Tell me about the history of Rome, Italy",
    "One story takes 2 days, and there are 20 stories"
]

for p in prompts:
    print(f"Prompt: {p}")
    print(f"Response: {routing_agent.route_prompt(p)}")
    print("-" * 30)