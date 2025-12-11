from typing import Dict, Any, List
from services.embedding_service import embedding_service
from services.qdrant_service import qdrant_service
from services.llm_service import llm_service
from supabase import create_client, Client
from config.settings import settings
from utils.logger import logger
from uuid import uuid4
import json
from models.schemas import SubjectiveQuestion, SubjectiveEvaluationResult

class EvaluationService:
    def __init__(self):
        self.client: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY
        )
    
    async def generate_subjective_test(
        self,
        project_id: str,
        topic: str = None,
        num_questions: int = 3,
        selected_documents: List[str] = None
    ) -> Dict[str, Any]:
        """Generate subjective test from project documents"""
        try:
            # 1. Get relevant content
            logger.info(f"Retrieving content for subjective test on topic: {topic if topic else 'General'}")
            # Use a strategy similar to MCQ but maybe more focused on broad concepts for subjective q's
            content = await self._get_relevant_context(project_id, topic if topic else "key concepts summary", num_chunks=15, selected_documents=selected_documents)
            
            if not content:
                logger.warning("No specific content found, generating generic questions")
            
            # 2. Generate Questions using LLM
            prompt_topic = f"Topic: {topic}" if topic else "Topic: General Review"
            
            prompt = f"""Based on the following educational content, generate {num_questions} questions for a study guide.
{prompt_topic}

Content:
{content}

Requirements:
- Questions should be thought-provoking.
- **Use Markdown** in the questions.
- Provide a detailed **answer** for each question (also in Markdown).

Format your response as a **JSON array**:
[
  {{
    "id": 1,
    "question": "What is the primary function of...",
    "answer": "The primary function is **bolded term** because..."
  }}
]

Respond ONLY with the valid JSON array."""

            messages = [{"role": "user", "content": prompt}]
            response = await llm_service.chat_completion(messages, temperature=0.7, max_tokens=2000)
            
            # 3. Parse response
            questions_data = self._parse_json_response(response)
            if not questions_data:
                raise Exception("Failed to parse generated questions")
            
            # 4. Store in database
            test_id = str(uuid4())
            self.client.table("subjective_tests").insert({
                "id": test_id,
                "project_id": project_id,
                "topic": topic if topic else "General Quiz",
                "questions": json.dumps(questions_data)
            }).execute()
            
            logger.info(f"Created subjective test (Q&A) with ID: {test_id}")
            
            # Return to client
            client_questions = [
                SubjectiveQuestion(
                    id=q.get("id", i+1), 
                    question=q["question"],
                    answer=q.get("answer", q.get("model_answer", "Answer not available"))
                )
                for i, q in enumerate(questions_data)
            ]
            
            return {
                "test_id": test_id,
                "topic": topic,
                "questions": client_questions
            }
            
        except Exception as e:
            logger.error(f"Error generating subjective test: {str(e)}")
            raise

    async def submit_subjective_test(
        self,
        test_id: str,
        answers: Dict[int, str]
    ) -> Dict[str, Any]:
        """Evaluate submitted subjective test answers"""
        try:
            # 1. Get test from database
            response = self.client.table("subjective_tests").select("*").eq("id", test_id).execute()
            
            if not response.data:
                raise Exception("Test not found")
            
            test = response.data[0]
            questions_data = json.loads(test["questions"])
            
            # Map questions by ID for easy access
            questions_map = {q.get("id", i+1): q for i, q in enumerate(questions_data)}
            
            evaluations = []
            total_score = 0
            max_score = 0
            
            # 2. Evaluate each answer
            for q_id_str, user_answer in answers.items():
                q_id = int(q_id_str)
                question_obj = questions_map.get(q_id)
                
                if not question_obj:
                    continue
                
                question_text = question_obj["question"]
                model_answer = question_obj.get("model_answer", "")
                
                # AI Evaluation
                eval_result = await self._evaluate_single_answer_internal(
                    question_text, 
                    user_answer, 
                    model_answer
                )
                
                score = eval_result.get("score", 0)
                total_score += score
                max_score += 10 # Assuming 10 points per question
                
                evaluations.append(SubjectiveEvaluationResult(
                    question_id=q_id,
                    question=question_text,
                    user_answer=user_answer,
                    score=score,
                    feedback=eval_result.get("feedback", "No feedback provided"),
                    suggestions=eval_result.get("suggestions", []),
                    model_answer=model_answer # Optional to show back
                ))
            
            percentage = (total_score / max_score * 100) if max_score > 0 else 0
            
            # 3. Update test record with results
            results_data = {
                "total_score": total_score,
                "max_score": max_score,
                "percentage": percentage,
                "evaluations": [e.model_dump() for e in evaluations]
            }
            
            self.client.table("subjective_tests").update({
                "results": json.dumps(results_data)
            }).eq("id", test_id).execute()
            
            return {
                "test_id": test_id,
                **results_data
            }
            
        except Exception as e:
            logger.error(f"Error submitting subjective test: {str(e)}")
            raise

    async def _evaluate_single_answer_internal(self, question: str, user_answer: str, model_answer: str) -> Dict[str, Any]:
        """Helper to evaluate a single answer against a model answer"""
        try:
            prompt = f"""Evaluate the following student answer.
            
Question: {question}
Model Answer / Key Points: {model_answer}

Student Answer:
{user_answer}

Evaluate based on:
1. Correctness (does it match the key points?)
2. Completeness
3. Clarity

Provide output in **valid JSON**:
{{
  "score": <integer 0-10>,
  "feedback": "**Markdown** feedback...",
  "suggestions": ["Suggestion 1", "Suggestion 2"]
}}

Respond ONLY with JSON.
"""
            messages = [{"role": "user", "content": prompt}]
            response = await llm_service.chat_completion(messages, temperature=0.3)
            result = self._parse_json_response(response)
            if not result:
                raise ValueError("Failed to parse evaluation response")
            return result
        except Exception as e:
            logger.error(f"Error in single eval: {e}")
            return {"score": 0, "feedback": "Error evaluating", "suggestions": []}

    async def evaluate_answer(
        self,
        project_id: str,
        user_id: str,
        question: str,
        user_answer: str
    ) -> Dict[str, Any]:
        """Evaluate subjective answer using AI (Single Question Mode)"""
        try:
            # 1. Get relevant context from documents
            logger.info(f"Retrieving context for question: {question[:50]}...")
            context = await self._get_relevant_context(project_id, question)
            
            # 2. Build evaluation prompt
            prompt = f"""You are an educational evaluator. Evaluate the following student answer based on the provided context.

Question: {question}

Reference Context from Documents:
{context}

Student Answer:
{user_answer}

Provide a comprehensive evaluation in JSON format:
{{
  "score": <number 0-10>,
  "feedback": "**Markdown** feedback...",
  "suggestions": ["<suggestion 1>", "<suggestion 2>"],
  "key_points_covered": ["<point 1>", "<point 2>"],
  "key_points_missed": ["<point 1>", "<point 2>"]
}}

Be constructive and specific in your feedback. Respond ONLY with the JSON object."""

            messages = [{"role": "user", "content": prompt}]
            response = await llm_service.chat_completion(messages, temperature=0.3, max_tokens=1500)
            
            # 3. Parse evaluation
            evaluation = self._parse_json_response(response)
            
            if not evaluation:
                raise Exception("Failed to parse evaluation response")
            
            # 4. Store in database
            evaluation_id = str(uuid4())
            self.client.table("answer_evaluations").insert({
                "id": evaluation_id,
                "project_id": project_id,
                "user_id": user_id,
                "question": question,
                "user_answer": user_answer,
                "ai_feedback": json.dumps(evaluation),
                "score": evaluation.get("score", 0)
            }).execute()
            
            logger.info(f"Created evaluation with ID: {evaluation_id}")
            
            return {
                "evaluation_id": evaluation_id,
                "question": question,
                "user_answer": user_answer,
                **evaluation
            }
            
        except Exception as e:
            logger.error(f"Error evaluating answer: {str(e)}")
            raise
    
    async def _get_relevant_context(
        self,
        project_id: str,
        query_text: str,
        num_chunks: int = 5,
        selected_documents: List[str] = None
    ) -> str:
        """Retrieve relevant context for evaluation using Query Expansion"""
        try:
            collection_name = f"project_{project_id}"
            
            # Simple Query Expansion for now
            queries = [query_text]
            
            # If query is very short, expand it? 
            # For evaluation, the query IS the question, which is usually specific enough.
            # But let's add a variation to be safe.
            expansion_prompt = f"""Generate 2 distinct search queries to find information relevant to answering this question: "{query_text}"
            Return only the queries separated by newlines."""
            
            try:
                expansion_response = await llm_service.chat_completion([{"role": "user", "content": expansion_prompt}], temperature=0.5)
                queries.extend([q.strip() for q in expansion_response.split('\n') if q.strip()])
            except:
                pass
            
            all_hits = []
            seen_texts = set()
            
            limit_per_query = max(2, num_chunks // len(queries))

            for q in queries:
                # Generate embedding for question/topic
                query_embedding = await embedding_service.generate_embedding(q)
                
                # Search in Qdrant
                results = await qdrant_service.search(
                    collection_name=collection_name,
                    query_vector=query_embedding,
                    limit=limit_per_query,
                    filter_conditions={"document_ids": selected_documents} if selected_documents else None
                )
                
                for hit in results:
                    if hit["text"] not in seen_texts:
                        all_hits.append(hit)
                        seen_texts.add(hit["text"])
            
            # Combine chunks
            if not all_hits:
                return ""
            
            context = "\n\n".join([hit["text"] for hit in all_hits[:num_chunks]])
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return ""
    
    def _parse_json_response(self, response: str) -> Any:
        """Parse JSON from LLM response"""
        try:
            # Try to find JSON structure
            start_arr = response.find('[')
            start_obj = response.find('{')
            
            start = -1
            if start_arr != -1 and (start_obj == -1 or start_arr < start_obj):
                start = start_arr
                end = response.rfind(']') + 1
            elif start_obj != -1:
                start = start_obj
                end = response.rfind('}') + 1
            
            if start == -1 or end == 0:
                logger.warning("No JSON found in response")
                return None
            
            json_str = response[start:end]
            # strict=False allows control characters like newlines in strings
            return json.loads(json_str, strict=False)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {str(e)}")
            # Try to sanitize common issues if strict=False didn't work
            try:
                # Fallback: remove non-printable characters or handle unescaped quotes if possible
                # For now, just logging and returning None to be handled by caller
                pass
            except:
                pass
            return None
    
    async def get_evaluation(self, evaluation_id: str) -> Dict[str, Any]:
        """Retrieve evaluation by ID"""
        try:
            response = self.client.table("answer_evaluations").select("*").eq(
                "id", evaluation_id
            ).execute()
            
            if not response.data:
                raise Exception("Evaluation not found")
            
            eval_data = response.data[0]
            ai_feedback = json.loads(eval_data["ai_feedback"])
            
            return {
                "evaluation_id": eval_data["id"],
                "question": eval_data["question"],
                "user_answer": eval_data["user_answer"],
                **ai_feedback
            }
            
        except Exception as e:
            logger.error(f"Error retrieving evaluation: {str(e)}")
            raise

evaluation_service = EvaluationService()
