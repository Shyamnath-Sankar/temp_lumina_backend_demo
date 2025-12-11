import json
import asyncio
from typing import List, Dict, Any, Union
from services.embedding_service import embedding_service
from services.qdrant_service import qdrant_service
from services.llm_service import llm_service
from db.client import supabase_client
from config.settings import settings
from utils.logger import logger
from uuid import uuid4

class MCQService:
    def __init__(self):
        self.client = supabase_client
    
    async def generate_document_topics(self, project_id: str, document_id: str) -> List[str]:
        """Generate topics for a single document"""
        try:
            collection_name = f"project_{project_id}"
            chunks = await qdrant_service.get_initial_chunks(
                collection_name=collection_name,
                document_id=document_id,
                limit=20  # Increased to 20 to ensure we capture Table of Contents past front matter
            )
            
            if not chunks:
                return []
            
            text = "\n".join(chunks)
            
            prompt = f"""Analyze the following text from the beginning of a document.
Extract a COMPREHENSIVE list of the main **Table of Contents** entries, Chapters, or Key Topics.
**Rules:**
1. Ignore "Front Matter" (e.g., Preface, Foreword, Copyright, Acknowledgements, List of Abbreviations).
2. Ignore "Back Matter" (e.g., Index, Appendix) unless they are substantial.
3. Focus on the **Core Educational Content**.
4. Capture hierarchical chapter titles if present (e.g., "Part III: Fundamental Rights").
5. Return ONLY a JSON array of strings.
6. Target **15-25 topics** to ensure good coverage.

Text:
{text[:15000]}
"""
            messages = [{"role": "user", "content": prompt}]
            response = await llm_service.chat_completion(messages, temperature=0.5)
            
            topics = []
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end != 0:
                try:
                    loaded = json.loads(response[start:end])
                    if isinstance(loaded, list):
                        topics = loaded
                except: pass
            
            # Save to document
            if topics:
                self.client.table("documents").update({"topics": topics}).eq("id", document_id).execute()
                logger.info(f"Generated topics for doc {document_id}")
            
            return topics
        except Exception as e:
            logger.error(f"Error doc topics: {e}")
            return []

    async def get_topics(self, project_id: str) -> Dict[str, Any]:
        """Get aggregated topics AND per-document topics"""
        try:
            # Get all docs with their topics
            response = self.client.table("documents").select("id, topics").eq("project_id", project_id).eq("upload_status", "completed").execute()
            documents = response.data or []
            
            all_topics = set()
            by_doc = {}
            docs_needing_generation = []
            
            for doc in documents:
                doc_topics = doc.get("topics") or []
                by_doc[doc["id"]] = doc_topics
                
                if doc_topics:
                    for t in doc_topics:
                        all_topics.add(t)
                else:
                    docs_needing_generation.append(doc["id"])
            
            # Lazy generation for docs missing topics
            if docs_needing_generation:
                logger.info(f"Generating topics for {len(docs_needing_generation)} documents...")
                tasks = [self.generate_document_topics(project_id, doc_id) for doc_id in docs_needing_generation]
                # Run concurrently
                results = await asyncio.gather(*tasks)
                
                # Update our local maps with results
                for i, doc_id in enumerate(docs_needing_generation):
                    t_list = results[i]
                    by_doc[doc_id] = t_list
                    for t in t_list:
                        all_topics.add(t)
            
            return {
                "all": sorted(list(all_topics)),
                "by_doc": by_doc
            }
            
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            return {"all": [], "by_doc": {}}

    async def generate_mcq(
        self,
        project_id: str,
        topic: str = None,
        num_questions: int = 5,
        selected_documents: List[str] = None
    ) -> Dict[str, Any]:
        """Generate MCQ test from project documents"""
        try:
            # 1. Get relevant content for the chapter/topic
            logger.info(f"Retrieving content for topic: {topic if topic else 'General'}")
            # Increased chunks for better context
            content = await self._get_context_content(project_id, topic, num_chunks=15, selected_documents=selected_documents)
            
            if not content:
                # Fallback if no specific content found
                logger.warning("No specific content found, generating generic questions")
            
            # 2. Generate MCQs using LLM
            prompt_topic = f"Topic: {topic}" if topic else "Topic: General Review"
            
            prompt = f"""Based on the following educational content, generate {num_questions} multiple-choice questions.
{prompt_topic}

Content:
{content}

Requirements:
- Questions should test deep understanding and critical thinking.
- Each question should have 4 options (A, B, C, D).
- Only one option should be correct.
- **Use Markdown** for the question text and explanation (e.g., bold keywords, code blocks if relevant).
- Include a clear and detailed explanation for the correct answer.

Format your response as a **JSON array** with this structure:
[
  {{
    "question": "**Question text** here?",
    "options": [
      {{"option": "A", "text": "First option"}},
      {{"option": "B", "text": "Second option"}},
      {{"option": "C", "text": "Third option"}},
      {{"option": "D", "text": "Fourth option"}}
    ],
    "correct_answer": "A",
    "explanation": "Explanation of why **A** is correct..."
  }}
]

Respond ONLY with the valid JSON array. Do not add any markdown formatting (like ```json) around the response."""

            messages = [{"role": "user", "content": prompt}]
            response = await llm_service.chat_completion(messages, temperature=0.7, max_tokens=3000)
            
            # 3. Parse response
            questions = self._parse_mcq_response(response)
            
            if not questions:
                raise Exception("Failed to parse MCQ questions from response")
            
            # 4. Store in database
            test_id = str(uuid4())
            self.client.table("mcq_tests").insert({
                "id": test_id,
                "project_id": project_id,
                "chapter_name": topic if topic else "General Quiz",
                "questions": json.dumps(questions)
            }).execute()
            
            logger.info(f"Created MCQ test with ID: {test_id}")
            
            return {
                "test_id": test_id,
                "topic": topic,
                "questions": questions
            }
            
        except Exception as e:
            logger.error(f"Error generating MCQ: {str(e)}")
            raise
    
    async def _get_context_content(
        self,
        project_id: str,
        topic: str = None,
        num_chunks: int = 15,
        selected_documents: List[str] = None
    ) -> str:
        """Retrieve relevant content using Query Expansion (Multi-Query)"""
        try:
            collection_name = f"project_{project_id}"
            
            # 1. Generate variations of the query (Query Expansion)
            # If a topic is provided, we ask the LLM to generate sub-topics or related questions to broaden search.
            queries = [topic] if topic else ["important concepts", "summary", "key definitions"]
            
            if topic:
                expansion_prompt = f"""You are an AI assistant helping to search a vector database. 
Generate 3 alternative search queries or related sub-topics for the topic: "{topic}" 
to ensure we find all relevant information in an educational textbook.
Return only the 3 queries separated by newlines."""
                
                try:
                    expansion_response = await llm_service.chat_completion([{"role": "user", "content": expansion_prompt}], temperature=0.7)
                    queries.extend([q.strip() for q in expansion_response.split('\n') if q.strip()])
                except Exception as e:
                    logger.warning(f"Query expansion failed: {e}")

            logger.info(f"Generated {len(queries)} search queries: {queries}")

            # 2. Perform Search for each query
            all_hits = []
            seen_texts = set()
            
            # Distribute the chunk limit across queries, but ensure at least 3 per query
            limit_per_query = max(3, num_chunks // len(queries))
            
            for query_text in queries:
                query_embedding = await embedding_service.generate_embedding(query_text)
                
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
            
            # 3. Sort and Deduplicate (Implicitly done by set, but maybe we want to re-rank? 
            # For now, simple concatenation is sufficient for context window)
            
            # Limit total content size roughly
            final_chunks = [hit["text"] for hit in all_hits][:num_chunks]
            
            content = "\n\n".join(final_chunks)
            return content
            
        except Exception as e:
            logger.error(f"Error retrieving context content: {str(e)}")
            return ""
    
    def _parse_mcq_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract MCQ questions"""
        try:
            # Try to find JSON array in response
            start = response.find('[')
            end = response.rfind(']') + 1
            
            if start == -1 or end == 0:
                logger.warning("No JSON array found in response")
                return []
            
            json_str = response[start:end]
            questions = json.loads(json_str)
            
            # Validate structure
            for q in questions:
                if not all(key in q for key in ["question", "options", "correct_answer", "explanation"]):
                    logger.warning("Invalid question structure")
                    return []
            
            return questions
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {str(e)}")
            return []
    
    async def submit_test(
        self,
        test_id: str,
        answers: Dict[int, str]
    ) -> Dict[str, Any]:
        """Evaluate submitted test answers"""
        try:
            # 1. Get test from database
            response = self.client.table("mcq_tests").select("*").eq("id", test_id).execute()
            
            if not response.data:
                raise Exception("Test not found")
            
            test = response.data[0]
            questions = json.loads(test["questions"])
            
            # 2. Evaluate answers
            score = 0
            total = len(questions)
            feedback = []
            
            for i, question in enumerate(questions):
                user_answer = answers.get(i)
                correct_answer = question["correct_answer"]
                
                is_correct = user_answer == correct_answer
                if is_correct:
                    score += 1
                
                feedback.append({
                    "question_number": i + 1,
                    "question": question["question"],
                    "user_answer": user_answer,
                    "correct_answer": correct_answer,
                    "is_correct": is_correct,
                    "explanation": question["explanation"]
                })
            
            percentage = (score / total * 100) if total > 0 else 0
            
            return {
                "score": score,
                "total": total,
                "percentage": round(percentage, 2),
                "feedback": feedback
            }
            
        except Exception as e:
            logger.error(f"Error submitting test: {str(e)}")
            raise

mcq_service = MCQService()