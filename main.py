import asyncio
import os
import json
import requests
import faiss
import numpy as np
import sqlite3
import streamlit as st
import random

from dataclasses import dataclass, field, asdict
from typing import Annotated, Optional, TypedDict, List, Any, Dict

from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.functions import KernelArguments, kernel_function
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import OpenAIChatPromptExecutionSettings
from semantic_kernel.contents import ChatMessageContent, FunctionCallContent, FunctionResultContent, TextContent
from sentence_transformers import SentenceTransformer

api_key = st.secrets["api_keys"]["OPENAI_API_KEY"]

# --- Instructions
AGENT_INSTRUCTIONS = """You are an expert insurance claims consultant. Your name if asked is 'ICA'.  Your responsibilities include:

1. Processing customer claims from First Notification of Loss (FNOL)
2. Understanding coverage, terms and exclusions as well as communicating reasoning to customer
3. Understanding liability
4. Managing fraud risk in claims
5. Determining the likely cost and handling route
6. Ensure consumer duty requirements

If a claim document has been uploaded, use StructureClaimData to structure its contents and use output for any function that takes a claim_data parameter.
If a response is drafted, consumer duty should be checked.
"""

@dataclass
class AgentMessage:
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    function_response: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentResponse:
    messages: List[AgentMessage]
    thread: ChatHistoryAgentThread
    metrics: Dict[str, Any] = field(default_factory=dict)

class RiskEvaluator:
    @kernel_function(description="Evaluate the risk of a claim using a risk model.")
    async def assess_risk(
        self,
        claim_data: Annotated[dict, "Structured claim data with fields like incident_type and description."]
    ) -> dict:
        name = claim_data.get("claimant_name", "Jane Doe")
        incident = claim_data.get("incident_type", "flood")
        desc = claim_data.get("incident_description", "flooded kitchen")

        # Random risk level assignment
        risk_level = random.choices(
            ["Low", "Medium", "High"],
            weights=[0.5, 0.3, 0.2],
            k=1
        )[0]

        rationale = f"Risk for claimant {name} involving incident '{incident}' assessed as {risk_level} based on our model."

        return {
            "risk_score": risk_level,
            "explanation": rationale
        }

class ClaimSizeEstimator:
    @kernel_function(description="Estimate the financial cost of a claim in GBP.")
    async def estimate_size(
        self,
        claim_data: Annotated[dict, "Structured insurance claim data."]
    ) -> dict:
        incident_type = claim_data.get("incident_type", "").lower()
        base = {
            "fire": 15000,
            "flood": 10000,
            "theft": 5000,
            "collision": 8000
        }.get(incident_type, 3000)

        # Add variance for detail
        variance = random.randint(-1000, 3000)
        estimate = max(1000, base + variance)

        return {
            "estimated_claim_value": estimate,
            "currency": "GBP"
        }

class DataCollector:
    def __init__(self, kernel: Kernel):
        self.conn = sqlite3.connect(":memory:")
        self.kernel = kernel
        self._setup_db()

    def _setup_db(self):
        cursor = self.conn.cursor()
        
        # Create customers table
        cursor.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            dob TEXT,
            address TEXT,
            email TEXT,
            phone TEXT,
            created_date TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Create policies table with foreign key to customers
        cursor.execute("""
        CREATE TABLE policies (
            policy_number TEXT PRIMARY KEY,
            customer_id INTEGER,
            coverage TEXT,
            coverage_limit INTEGER,
            deductible INTEGER,
            start_date TEXT,
            end_date TEXT,
            status TEXT CHECK(status IN ('active', 'expired', 'cancelled')),
            FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
        )
        """)

        # Create historic_claims table
        cursor.execute("""
        CREATE TABLE historic_claims (
            claim_id INTEGER PRIMARY KEY,
            policy_number TEXT,
            incident_type TEXT,
            incident_date TEXT,
            claim_amount DECIMAL,
            status TEXT CHECK(status IN ('approved', 'rejected', 'pending')),
            decision_date TEXT,
            description TEXT,
            FOREIGN KEY (policy_number) REFERENCES policies (policy_number)
        )
        """)

        # Insert sample customer data
        customers = [
            (1, "Jane Doe", "1985-03-10", "123 Main St, London", "jane.doe@email.com", "+44 20 1234 5678"),
            (2, "John Smith", "1990-07-22", "456 Park Ave, Manchester", "john.smith@email.com", "+44 20 8765 4321"),
            (3, "Sarah Wilson", "1978-11-15", "789 High St, Birmingham", "sarah.w@email.com", "+44 20 5555 1234")
        ]
        cursor.executemany(
            "INSERT INTO customers (customer_id, name, dob, address, email, phone) VALUES (?, ?, ?, ?, ?, ?)",
            customers
        )

        # Insert sample policy data
        policies = [
            ("POL123456", 1, "Theft,Fire,Flood", 20000, 500, "2025-01-01", "2026-01-01", "active"),
            ("POL654321", 2, "Flood,Fire", 30000, 1000, "2025-02-01", "2026-02-01", "active"),
            ("POL789012", 3, "Theft,Fire,Flood", 50000, 750, "2024-12-01", "2025-12-01", "active")
        ]
        cursor.executemany(
            "INSERT INTO policies VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            policies
        )

        # Insert sample claims history
        claims = [
            (1, "POL123456", "Theft", "2025-02-15", 5000.00, "approved", "2025-02-20", "Laptop stolen from office"),
            (2, "POL654321", "Fire", "2025-03-01", 15000.00, "approved", "2025-03-10", "Kitchen fire damage"),
            (3, "POL123456", "Collision", "2025-04-01", 2000.00, "rejected", "2025-04-05", "Minor vehicle damage"),
            (4, "POL789012", "Flood", "2025-01-15", 25000.00, "approved", "2025-01-25", "Basement flooding")
        ]
        cursor.executemany(
            "INSERT INTO historic_claims VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            claims
        )

        self.conn.commit()


    @kernel_function(description="Validate a policy using policy number or claimant info against internal database.")
    async def get_user_policy(
        self,
        claim_data: Annotated[dict, "Structured claim object containing at least claimant_name or policy_number."]
    ) -> dict:
        cursor = self.conn.cursor()
        if claim_data.get("policy_number"):
            cursor.execute("""
                SELECT p.*, c.name, c.dob, c.address
                FROM policies p
                JOIN customers c ON p.customer_id = c.customer_id
                WHERE p.policy_number = ?
            """, (claim_data["policy_number"],))
        else:
            cursor.execute("""
                SELECT p.*, c.name, c.dob, c.address
                FROM policies p
                JOIN customers c ON p.customer_id = c.customer_id
                WHERE c.name = ?
            """, (claim_data.get("claimant_name", ""),))

        row = cursor.fetchone()
        if not row:
            return {"error": "Policy not found"}

        return {
            "policy_number": row[0],
            "claimant_name": row[8],  # from joined customers table
            "dob": row[9],
            "address": row[10],
            "covered_incidents": row[2].split(","),
            "coverage_limit": row[3],
            "deductible": row[4],
            "policy_status": row[7]
        }

    @kernel_function(description="Retrieve claim history for a customer using policy number or claimant name.")
    async def get_claim_history(
        self,
        claim_data: Annotated[dict, "Structured claim object containing at least claimant_name or policy_number."]
    ) -> dict:
        cursor = self.conn.cursor()
        
        if claim_data.get("policy_number"):
            cursor.execute("""
                SELECT 
                    h.claim_id,
                    h.incident_type,
                    h.incident_date,
                    h.claim_amount,
                    h.status,
                    h.decision_date,
                    h.description,
                    c.name as claimant_name,
                    p.policy_number
                FROM historic_claims h
                JOIN policies p ON h.policy_number = p.policy_number
                JOIN customers c ON p.customer_id = c.customer_id
                WHERE h.policy_number = ?
                ORDER BY h.incident_date DESC
            """, (claim_data["policy_number"],))
        else:
            cursor.execute("""
                SELECT 
                    h.claim_id,
                    h.incident_type,
                    h.incident_date,
                    h.claim_amount,
                    h.status,
                    h.decision_date,
                    h.description,
                    c.name as claimant_name,
                    p.policy_number
                FROM historic_claims h
                JOIN policies p ON h.policy_number = p.policy_number
                JOIN customers c ON p.customer_id = c.customer_id
                WHERE c.name = ?
                ORDER BY h.incident_date DESC
            """, (claim_data.get("claimant_name", ""),))

        rows = cursor.fetchall()
        if not rows:
            return {"error": "No claim history found"}

        claims = []
        for row in rows:
            claims.append({
                "claim_id": row[0],
                "incident_type": row[1],
                "incident_date": row[2],
                "claim_amount": float(row[3]),
                "status": row[4],
                "decision_date": row[5],
                "description": row[6],
                "claimant_name": row[7],
                "policy_number": row[8]
            })

        return {
            "claimant_name": rows[0][7],
            "policy_number": rows[0][8],
            "total_claims": len(claims),
            "total_approved_amount": sum(c["claim_amount"] for c in claims if c["status"] == "approved"),
            "claims": claims
        }

class VectorMemoryRAGPlugin:
    def __init__(self):
        self.text_chunks = []
        self.index = None
        self.embeddings = SentenceTransformer("all-MiniLM-L6-v2")

    def add_document(self, doc_text: str, chunk_size: int = 500):
        self.text_chunks = [
            doc_text[i:i + chunk_size]
            for i in range(0, len(doc_text), chunk_size)
        ]
        vectors = self.embeddings.encode(self.text_chunks, convert_to_numpy=True)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

    @kernel_function(description="retrieve relevant chunks from uploaded claim documents.")
    async def retrieve_chunks(self, query: Annotated[str, "Query to summmarise / retrieve relevant claim information"]) -> str:
        if not self.index:
            return "No documents indexed yet."
        query_vec = self.embeddings.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vec, k=3)
        relevant_chunks = [self.text_chunks[i] for i in I[0] if i < len(self.text_chunks)]
        return "\n---\n".join(relevant_chunks)

class ConsumerDutyChecker:
    def __init__(self, kernel: Optional[Kernel] = None):
        self.kernel = kernel
    
    @kernel_function(
        description="Evaluate if an insurance claim decision meets UK consumer duty requirements",
        name="check_consumer_duty"
    )
    async def check_consumer_duty(
        self,
        decision_text: Annotated[str, "The claim decision and explanation"],
    ) -> dict:
        prompt = f"""
System: You are a UK Consumer Duty compliance expert specializing in insurance claims evaluation.
Your task is to analyze the following insurance claim decision and provide a DETAILED evaluation 
of FCA Consumer Duty compliance requirements.

IMPORTANT: You MUST provide specific, detailed notes for each requirement explaining exactly why 
it passed or failed. Do not leave any fields empty.

For the decision text below:
\"\"\"{decision_text}\"\"\"

Evaluate each requirement and explain your reasoning:

1. Clear Communication
- Check for clear, simple language
- Identify any technical jargon
- Assess readability level

2. Fair Treatment
- Evaluate if the decision process was fair
- Check if all relevant factors were considered
- Assess if the outcome is justified

3. Transparent Reasoning
- Verify if decision rationale is clearly explained
- Check if policy terms are referenced correctly
- Ensure all key points are addressed

4. Consumer Understanding
- Assess if an average customer would understand
- Check if next steps are clearly explained
- Verify if important points are emphasized

5. Vulnerable Customer Considerations
- Check if potential vulnerabilities are considered
- Assess if additional support is offered
- Evaluate accessibility of communication

You MUST return a fully populated JSON response with detailed notes for each section:

{{
    "meets_requirements": true/false,
    "checklist": {{
        "clear_communication": {{
            "passed": true/false,
            "notes": "DETAILED explanation of why it passed or failed"
        }},
        "fair_treatment": {{
            "passed": true/false,
            "notes": "DETAILED explanation of why it passed or failed"
        }},
        "transparent_reasoning": {{
            "passed": true/false,
            "notes": "DETAILED explanation of why it passed or failed"
        }},
        "consumer_understanding": {{
            "passed": true/false,
            "notes": "DETAILED explanation of why it passed or failed"
        }},
        "vulnerable_customers": {{
            "passed": true/false,
            "notes": "DETAILED explanation of why it passed or failed"
        }}
    }},
    "improvement_suggestions": [
        "Specific actionable improvements needed"
    ],
    "risk_flags": [
        "Specific compliance risks that need addressing"
    ]
}}

Respond ONLY with a valid JSON object. Do not include any text before or after the JSON.
"""
        completion = await self.kernel.invoke_prompt(prompt)
        if hasattr(completion, "result"):
            return str(completion.result).strip()
        return str(completion).strip()

class StructureClaimData:
    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    @kernel_function(description="Return a JSON containing structured claim_data, use before calling other plugins")
    async def StructureClaimData(self, claim_text: Annotated[str, "The unstructured claim_text string input"]) -> str:
        prompt = f"""
Extract the following fields from the text below. If a field is not present, leave it blank.

Required fields (as JSON):
{{
    "claimant_name": "",
    "date_of_birth": "",
    "address": "",
    "incident_type": "",
    "incident_date": "",
    "incident_description": "",
    "policy_number": ""
}}

Text:
\"\"\"{claim_text}\"\"\"

Respond ONLY with a valid JSON object. Do not include any text before or after the JSON.
"""
        completion = await self.kernel.invoke_prompt(prompt)
        if hasattr(completion, "result"):
            return str(completion.result).strip()
        return str(completion).strip()

# --- Main async entrypoint
async def main(
    user_input: str, 
    thread: Optional[ChatHistoryAgentThread] = None, 
    claim_text: Optional[str] = None
) -> AgentResponse:
    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(
        ai_model_id="gpt-4.1-mini",
        api_key=api_key,
    ))

    messages: List[AgentMessage] = []

    # 👉 Keep RAG setup for policy lookup
    vector_memory_rag = VectorMemoryRAGPlugin()
    if claim_text:
        vector_memory_rag.add_document(claim_text)

    # --- Register plugins
    kernel.add_plugin(DataCollector(kernel), plugin_name="collector")    
    kernel.add_plugin(vector_memory_rag, plugin_name="VectorMemoryRAG")
    kernel.add_plugin(RiskEvaluator(), plugin_name="RiskModel")
    kernel.add_plugin(ClaimSizeEstimator(), plugin_name="ClaimEstimator")
    kernel.add_plugin(ConsumerDutyChecker(kernel), plugin_name="ConsumerDuty")
    kernel.add_plugin(StructureClaimData(kernel), plugin_name="StructureClaimData")

    
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="ICA",
        instructions=AGENT_INSTRUCTIONS,
        arguments=KernelArguments(
            settings=OpenAIChatPromptExecutionSettings(
                temperature=0.5,
                top_p=0.95,
                function_choice_behavior=FunctionChoiceBehavior.Auto()
            )
        )
    )

    intermediate_steps: List[AgentMessage] = []
    metrics = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "steps": 0
    }

    messages.append(AgentMessage(role="user", content=user_input))

    async def handle_intermediate(message: ChatMessageContent):
        for item in message.items:
            if isinstance(item, FunctionCallContent):
                intermediate_steps.append(AgentMessage(
                    role="function_call",
                    name=item.name,
                    function_call={"name": item.name, "arguments": item.arguments},
                ))
            elif isinstance(item, FunctionResultContent):
                intermediate_steps.append(AgentMessage(
                    role="function_response",
                    name=item.name,
                    function_response=item.result,
                ))
            else:
                intermediate_steps.append(AgentMessage(
                    role=message.role.value,
                    content=message.content
                ))

    async for response in agent.invoke(
        messages=user_input,
        thread=thread,
        on_intermediate_message=handle_intermediate
    ):
        message = AgentMessage(
            role=response.role,
            content=response.content,
            name=getattr(response, "name", None)
        )

        if hasattr(response, "function_call") and response.function_call:
            message.function_call = {
                "name": response.function_call.name,
                "arguments": response.function_call.arguments
            }

        if hasattr(response, "function_response") and response.function_response:
            message.function_response = response.function_response

        if hasattr(response, "metadata") and response.metadata:
            message.metadata = response.metadata
            if "usage" in message.metadata:
                usage = message.metadata["usage"]
                metrics["prompt_tokens"] += usage.prompt_tokens
                metrics["completion_tokens"] += usage.completion_tokens
                metrics["total_tokens"] = metrics["prompt_tokens"] + metrics["completion_tokens"]

        messages.extend(intermediate_steps)
        metrics["steps"] += len(intermediate_steps)
        intermediate_steps.clear()
        messages.append(message)
        metrics["steps"] += 1
        thread = response.thread

    return AgentResponse(
        messages=messages,
        thread=thread,
        metrics=metrics
    )

# --- Debug runner
if __name__ == "__main__":
    async def test():
        response = await main("produce a bid to sell a new AI note taking product to Microsoft")
        for msg in response.messages:
            print(f"\n[{msg.role}]")
            if msg.content:
                print(msg.content.strip())
            if msg.function_call:
                print(f"Function: {msg.function_call['name']}")
            if msg.function_response:
                print(f"Result: {str(msg.function_response)[:100]}...")
        print(f"\nMetrics: {response.metrics}")

    asyncio.run(test())
