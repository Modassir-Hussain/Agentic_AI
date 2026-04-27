import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime
from openai import OpenAI

# 1. DirectPromptAgent
class DirectPromptAgent:
    def __init__(self, openai_api_key):
        # TODO: 2 - Store the OpenAI API key
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        client = OpenAI(base_url="https://openai.vocareum.com/v1",api_key=self.openai_api_key)
        # TODO: 3 & 4 - Use gpt-3.5-turbo and pass prompt directly
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        # TODO: 5 - Return only textual content
        return response.choices[0].message.content

# 2. AugmentedPromptAgent
class AugmentedPromptAgent:
    def __init__(self, openai_api_key, persona):
        # TODO: 1 - Create an attribute for persona
        self.persona = persona
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        client = OpenAI(base_url="https://openai.vocareum.com/v1",api_key=self.openai_api_key)
        # TODO: 2 & 3 - System prompt with persona and context reset
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a {self.persona}. Forget all previous conversational context."},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        # TODO: 4 - Return textual content
        return response.choices[0].message.content

# 3. KnowledgeAugmentedPromptAgent
class KnowledgeAugmentedPromptAgent:
    def __init__(self, openai_api_key, persona, knowledge):
        self.persona = persona
        # TODO: 1 - Store agent's knowledge
        self.knowledge = knowledge
        self.openai_api_key = openai_api_key

    def respond(self, input_text):
        client = OpenAI(base_url="https://openai.vocareum.com/v1",api_key=self.openai_api_key)
        # TODO: 2 & 3 - Construct grounded system message
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        f"You are {self.persona} knowledge-based assistant. Forget all previous context. "
                        f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge} "
                        f"Answer the prompt based on this knowledge, not your own."
                    )
                },
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content

# 5. EvaluationAgent
class EvaluationAgent:
    def __init__(self, openai_api_key, persona, evaluation_criteria, worker_agent, max_interactions):
        # TODO: 1 - Declare class attributes
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        client = OpenAI(base_url="https://openai.vocareum.com/v1",api_key=self.openai_api_key)
        prompt_to_evaluate = initial_prompt
        final_response = ""
        evaluation = ""
        iteration_count = 0

        # TODO: 2 - Interaction loop
        for i in range(self.max_interactions):
            iteration_count = i + 1
            print(f"\n--- Interaction {iteration_count} ---")

            # TODO: 3 - Obtain worker response
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            final_response = response_from_worker

            # TODO: 4 & 5 - Evaluator judges the response
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": f"You are {self.persona}."},
                          {"role": "user", "content": eval_prompt}],
                temperature=0
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            if evaluation.lower().startswith("yes"):
                print("Final solution accepted.")
                break
            else:
                # TODO: 6 - Generate correction instructions
                instruction_prompt = f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
                
                instr_resp = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": instruction_prompt}],
                    temperature=0
                )
                instructions = instr_resp.choices[0].message.content.strip()
                
                # Feedback loop for next iteration
                prompt_to_evaluate = (
                    f"The original prompt was: {initial_prompt}\n"
                    f"The response to that prompt was: {response_from_worker}\n"
                    f"It has been evaluated as incorrect.\n"
                    f"Make only these corrections, do not alter content validity: {instructions}"
                )
        
        # TODO: 7 - Return results dictionary
        return {
            "final_response": final_response,
            "evaluation": evaluation,
            "iterations": iteration_count
        }

# 6. RoutingAgent
class RoutingAgent:
    def __init__(self, openai_api_key, agents):
        self.openai_api_key = openai_api_key
        # TODO: 1 - Store agents
        self.agents = agents 

    def get_embedding(self, text):
        client = OpenAI(base_url="https://openai.vocareum.com/v1",api_key=self.openai_api_key)
        # TODO: 2 - Calculate embedding
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding 

    # TODO: 3 - Route user prompts
    def route_prompt(self, user_input):
        # TODO: 4 - Compute input embedding
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            # TODO: 5 - Compute agent description embedding
            agent_emb = self.get_embedding(agent['description'])
            
            # Cosine Similarity
            similarity = np.dot(input_emb, agent_emb) / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            
            # TODO: 6 - Selection logic
            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)

# 7. ActionPlanningAgent
class ActionPlanningAgent:
    def __init__(self, openai_api_key, knowledge):
        # TODO: 1 - Initialize attributes
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):
        # TODO: 2 & 3 - Instantiate client and call API
        client = OpenAI(base_url="https://openai.vocareum.com/v1",api_key=self.openai_api_key)
        sys_prompt = (
            f"You are an action planning agent. Using your knowledge, you extract from the user prompt "
            f"the steps requested to complete the action the user is asking for. You return the steps as a list. "
            f"Only return the steps in your knowledge. Forget any previous context. "
            f"This is your knowledge: {self.knowledge}"
        )
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        # TODO: 4 - Extract text
        response_text = response.choices[0].message.content.strip()

        # TODO: 5 - Clean and format steps
        # Splitting and filtering out empty lines or common markers
        raw_steps = response_text.split("\n")
        steps = [re.sub(r'^\d+\.\s*|-\s*', '', s).strip() for s in raw_steps if s.strip()]

        return steps

# 8. RAGKnowledgePromptAgent
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text):
        """
        Fetches the embedding vector for given text using OpenAI's embedding API.

        Parameters:
        text (str): Text to embed.

        Returns:
        list: The embedding vector.
        """
        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # def chunk_text(self, text):
    #     """
    #     Splits text into manageable chunks, attempting natural breaks.

    #     Parameters:
    #     text (str): Text to split into chunks.

    #     Returns:
    #     list: List of dictionaries containing chunk metadata.
    #     """
    #     separator = "\n"
    #     text = re.sub(r'\s+', ' ', text).strip()

    #     if len(text) <= self.chunk_size:
    #         return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

    #     chunks, start, chunk_id = [], 0, 0

    #     while start < len(text):
    #         end = min(start + self.chunk_size, len(text))
    #         if separator in text[start:end]:
    #             end = start + text[start:end].rindex(separator) + len(separator)

    #         chunks.append({
    #             "chunk_id": chunk_id,
    #             "text": text[start:end],
    #             "chunk_size": end - start,
    #             "start_char": start,
    #             "end_char": end
    #         })

    #         start = end - self.chunk_overlap
    #         chunk_id += 1

    #     with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
    #         writer.writeheader()
    #         for chunk in chunks:
    #             writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

    #     return chunks

    def chunk_text(self, text):
        """
        Splits text into manageable chunks, ensuring the loop always progresses to avoid MemoryErrors.
        """
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        separator = " " # Using a simple space as a natural break point

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            # 1. Calculate the potential end of this chunk
            end = min(start + self.chunk_size, len(text))
            
            # 2. Try to find a natural break point (space) to avoid cutting words
            if end < len(text):
                last_space = text.rfind(separator, start, end)
                if last_space != -1 and last_space > start:
                    end = last_space

            # 3. Save the chunk
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "chunk_size": len(chunk_text),
                    "start_char": start,
                    "end_char": end
                })
                chunk_id += 1

            # 4. --- THE CRITICAL SAFETY FIX ---
            # Calculate the next start point based on overlap
            next_start = end - self.chunk_overlap
            
            # If the overlap is too large and prevents progress, 
            # force the start to move to the current end.
            if next_start <= start:
                start = end
            else:
                start = next_start

        # 5. Save results to CSV
        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({"text": chunk["text"], "chunk_size": chunk["chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        """
        Calculates embeddings with progress tracking to prevent 'stuck' sessions.
        """
        print(f"--- [DEBUG] Reading chunks from: chunks-{self.unique_filename} ---")
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        
        total_chunks = len(df)
        print(f"--- [DEBUG] Total chunks to process: {total_chunks} ---")

        embeddings = []
        for i, text in enumerate(df['text']):
            print(f"--- [DEBUG] Processing chunk {i+1}/{total_chunks}... ", end="", flush=True)
            try:
                # Call the API for this specific chunk
                vector = self.get_embedding(text)
                embeddings.append(vector)
                print("Done. ---")
            except Exception as e:
                print(f"FAILED. Error: {e}")
                # Append a dummy vector so the DataFrame doesn't mismatch lengths
                embeddings.append([0.0] * 3072) 

        df['embeddings'] = embeddings
        
        print(f"--- [DEBUG] Saving to: embeddings-{self.unique_filename} ---")
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df['embeddings'] = df['embeddings'].apply(lambda x: np.array(eval(x)))
        df['similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        client = OpenAI(base_url="https://openai.vocareum.com/v1", api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context."},
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"}
            ],
            temperature=0
        )

        return response.choices[0].message.content
