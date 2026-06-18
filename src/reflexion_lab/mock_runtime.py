import os
import time
import json
from typing import Tuple
from google import genai
from google.genai import types
from pydantic import BaseModel
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv()

from .schemas import QAExample, JudgeResult, ReflectionEntry
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM

client = genai.Client() # Uses GEMINI_API_KEY from .env

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}

# Use gemini-2.5-flash as requested (can change to gemini-2.0-flash-lite-preview-02-05 if needed)
MODEL_NAME = "gemini-2.5-flash"

@retry(wait=wait_exponential(multiplier=1, min=5, max=60), stop=stop_after_attempt(10))
def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> Tuple[str, int, int]:
    start_time = time.time()
    
    context_str = "\n".join([f"Title: {c.title}\nText: {c.text}" for c in example.context])
    user_prompt = f"Question: {example.question}\nContext:\n{context_str}\n"
    
    if reflection_memory:
        user_prompt += "\nReflections from previous attempts:\n"
        for i, ref in enumerate(reflection_memory, 1):
            user_prompt += f"{i}. {ref}\n"
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=ACTOR_SYSTEM,
            temperature=0.0
        )
    )
    
    latency = int((time.time() - start_time) * 1000)
    tokens = response.usage_metadata.total_token_count if response.usage_metadata else 0
    answer = response.text.strip() if response.text else ""
    
    return answer, tokens, latency

@retry(wait=wait_exponential(multiplier=1, min=5, max=60), stop=stop_after_attempt(10))
def evaluator(example: QAExample, answer: str) -> Tuple[JudgeResult, int, int]:
    start_time = time.time()
    user_prompt = f"Question: {example.question}\nGold Answer: {example.gold_answer}\nPredicted Answer: {answer}"
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=EVALUATOR_SYSTEM,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=JudgeResult
        )
    )
    
    latency = int((time.time() - start_time) * 1000)
    tokens = response.usage_metadata.total_token_count if response.usage_metadata else 0
    
    try:
        data = json.loads(response.text)
        judge_result = JudgeResult(**data)
    except Exception as e:
        judge_result = JudgeResult(score=0, reason=f"Parse error: {e}")
        
    return judge_result, tokens, latency

@retry(wait=wait_exponential(multiplier=1, min=5, max=60), stop=stop_after_attempt(10))
def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> Tuple[ReflectionEntry, int, int]:
    start_time = time.time()
    user_prompt = f"Question: {example.question}\nFailure Reason: {judge.reason}"
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=REFLECTOR_SYSTEM,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=ReflectionEntry
        )
    )
    
    latency = int((time.time() - start_time) * 1000)
    tokens = response.usage_metadata.total_token_count if response.usage_metadata else 0
    
    try:
        data = json.loads(response.text)
        reflection_result = ReflectionEntry(**data)
        reflection_result.attempt_id = attempt_id
    except Exception as e:
        reflection_result = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Parse error",
            next_strategy="Try again"
        )
    
    return reflection_result, tokens, latency
