import os
import re
import json
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
client = chromadb.Client()
collection = client.get_or_create_collection("tickets")


class TicketDB:
    def __init__(self, collection):
        self.collection = collection

    def add_ticket(self, ticket: Dict[str, Any]):
        text = ticket["subject"] + " " + ticket["description"]
        embedding = embedding_model.encode(text).tolist()
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[ticket["id"]],
            metadatas=[ticket],
        )

    def search(self, query: str, n_results: int = 3):
        query_embedding = embedding_model.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        return results["documents"][0], results["metadatas"][0]


class TicketAgent:
    def __init__(self, groq_client: Groq, model: str = "llama-3.1-8b-instant"):
        self.groq_client = groq_client
        self.model = model

    def generate_reply(
        self, ticket: Dict[str, Any], retrieved_docs: List[str], retrieved_meta: List[Dict[str, Any]]
    ):
        context_intro = "\n\nSimilar tickets from knowledge base (for context):\n"
        for i, doc in enumerate(retrieved_docs):
            c = f"---Ticket {i+1}---\n{doc}\n" + (f"Meta: {retrieved_meta[i]}\n" if retrieved_meta else "")
            context_intro += c

        prompt = f"""You are an expert AI agent for ticket classification and support.
Return ONLY a valid JSON object with the following fields: 
category, subcategory, priority, solution, next_action.

Ticket:
Subject: {ticket['subject']}
Description: {ticket['description']}

{context_intro}
"""

        response = self.groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.1,
            max_tokens=512,
        )

        
        answer = response.choices[0].message.content.strip()

        
        match = re.search(r"\{.*\}", answer, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode failed: {e} | Raw answer: {answer}")

        # Fallback if parsing fails
        return {
            "category": "uncertain",
            "subcategory": "uncertain",
            "priority": "medium",
            "solution": "Sorry, unable to provide a solution automatically.",
            "next_action": "escalate",
        }

    def classify_and_solve(self, ticket: Dict[str, Any]):
        query = ticket["subject"] + " " + ticket["description"]
        similar_docs, meta = db.search(query)
        response = self.generate_reply(ticket, similar_docs, meta)
        return response


class TicketSystem:
    def __init__(self, agent: TicketAgent, db: TicketDB):
        self.agent = agent
        self.db = db

    def process_ticket(self, ticket: Dict[str, Any]):
        result = self.agent.classify_and_solve(ticket)
        # Save new ticket into database for future retrieval
        self.db.add_ticket(ticket)
        return result


# Load Groq API key from .env
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError(" Missing GROQ_API_KEY in .env file")

groq_client = Groq(api_key=groq_api_key)

# Initialize DB and Agent
db = TicketDB(collection)
agent = TicketAgent(groq_client)
system = TicketSystem(agent, db)

if __name__ == "__main__":
    ticket = {
        "id": "1",
        "subject": "Cannot connect to VPN",
        "description": "User is unable to establish a VPN connection after update.",
    }

    result = system.process_ticket(ticket)
    print("Processed Ticket Result:")
    print(json.dumps(result, indent=2))
