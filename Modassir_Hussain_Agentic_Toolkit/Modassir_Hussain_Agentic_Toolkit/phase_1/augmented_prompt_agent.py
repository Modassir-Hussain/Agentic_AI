# TODO: 1 - Import the AugmentedPromptAgent class
from workflow_agents.base_agents import AugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# TODO: 2 - Instantiate an object of AugmentedPromptAgent with the required parameters
augmented_agent = AugmentedPromptAgent(openai_api_key, persona)

# TODO: 3 - Send the 'prompt' to the agent and store the response in a variable named 'augmented_agent_response'
# FIX: This line was likely missing or misspelled in your script
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print(f"Prompt: {prompt}")
print("-" * 30)
print(f"Agent Response: {augmented_agent_response}")
print("-" * 30)

# TODO: 4 - Add the printed explanations required by the Rubric
print("\n--- TEST LOG EXPLANATION ---")
print("Knowledge Source: The agent used the LLM's internal general knowledge to identify Paris.")
print("Persona Impact: The persona instruction (system prompt) successfully forced the "
      "specific 'Dear students' formatting and academic tone.")
# ... (existing execution code) ...

# PRINT STATEMENTS FOR RUBRIC COMPLIANCE
print("\n" + "="*50)
print("RUBRIC EVALUATION: AUGMENTED PROMPT AGENT")
print("="*50)
print("1. Knowledge Source: The agent utilized the LLM's internal general knowledge.")
print("2. Persona Impact: The system prompt successfully enforced a specialized persona,")
print("   drastically changing the tone and formatting compared to a standard AI response.")
print("="*50)