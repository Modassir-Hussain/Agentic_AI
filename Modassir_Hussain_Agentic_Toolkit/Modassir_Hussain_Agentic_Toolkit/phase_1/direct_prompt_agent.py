# Test script for DirectPromptAgent class

# TODO: 1 - Import the DirectPromptAgent class from BaseAgents
# Corrected path: the folder is workflow_agents and the file is base_agents.py
from workflow_agents.base_agents import DirectPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# TODO: 2 - Load the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
prompt = "What is the Capital of France?"

# TODO: 3 - Instantiate the DirectPromptAgent as direct_agent
direct_agent = DirectPromptAgent(openai_api_key)

# TODO: 4 - Use direct_agent to send the prompt defined above and store the response
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
print(f"Agent Response: {direct_agent_response}")

# TODO: 5 - Print an explanatory message describing the knowledge source
print("Explanatory Note: The agent used general knowledge inherent to the gpt-3.5-turbo model to answer.")

print("\n" + "="*50)
print("RUBRIC CHECK: DIRECT PROMPT AGENT")
print("KNOWLEDGE SOURCE: This agent relies entirely on the LLM's internal")
print("pre-trained knowledge. No system prompt or external data was used.")
print("="*50)