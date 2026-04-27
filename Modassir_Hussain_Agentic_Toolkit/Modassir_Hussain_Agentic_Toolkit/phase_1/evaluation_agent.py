# TODO: 1 - Import EvaluationAgent and KnowledgeAugmentedPromptAgent classes
from workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
prompt = "What is the capital of France?"

# Parameters for the Knowledge Agent
# Note the "fake" knowledge here—we are testing if the agent follows the grounding!
knowledge_persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge_data = "The capital of France is London, not Paris"

# TODO: 2 - Instantiate the KnowledgeAugmentedPromptAgent
knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, knowledge_persona, knowledge_data)

# Parameters for the Evaluation Agent
evaluator_persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."

# TODO: 3 - Instantiate the EvaluationAgent with 10 interactions
evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=evaluator_persona,
    evaluation_criteria=evaluation_criteria,
    worker_agent=knowledge_agent,
    max_interactions=10
)

# TODO: 4 - Evaluate the prompt and print the response
result = evaluation_agent.evaluate(prompt)

print("\n--- FINAL EVALUATION RESULT ---")
print(f"Final Response: {result['final_response']}")
print(f"Total Iterations: {result['iterations']}")
print(f"Final Evaluation: {result['evaluation']}")