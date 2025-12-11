import os
import asyncio
from typing import Optional
from db.client import supabase_client
from config.settings import settings
from utils.file_parser import FileParser
from utils.text_chunker import TextChunker
from services.embedding_service import embedding_service
from services.qdrant_service import qdrant_service
from utils.logger import logger

class DocumentService:
    def __init__(self):
        self.client = supabase_client
        self.file_parser = FileParser()
        self.text_chunker = TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP
        )
    
    async def process_document(
        self,
        document_id: str,
        project_id: str,
        content: bytes,
        file_ext: str,
        filename: str
    ):
        """Process uploaded document: extract text, chunk, embed, and store"""
        try:
            # Update status to processing
            await self._update_document_status(document_id, "processing")

            loop = asyncio.get_running_loop()

            # 1. Extract text (Run in thread pool to avoid blocking)
            logger.info(f"Extracting text from {filename}")
            await self._update_document_status(document_id, "processing", "Extracting text...")
            text = await loop.run_in_executor(None, self.file_parser.extract_text, content, file_ext)
            
            if not text:
                await self._update_document_status(document_id, "failed", "Failed to extract text")
                return
            
            # 2. Chunk text (Run in thread pool to avoid blocking)
            # LangChain's splitter is CPU bound
            logger.info(f"Chunking text from {filename}")
            await self._update_document_status(document_id, "processing", "Chunking text...")
            chunks = await loop.run_in_executor(
                None, 
                lambda: self.text_chunker.chunk_text(text)
            )
            
            if not chunks:
                await self._update_document_status(document_id, "failed", "No chunks generated")
                return
            
            # 3. Generate embeddings (Async)
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            
            # 3. Create collection FIRST (to ensure it exists before upserting)
            collection_name = f"project_{project_id}"
            await qdrant_service.create_collection(collection_name)

            # 4. Generate embeddings and Upsert (Streamed)
            logger.info(f"Generating embeddings and upserting for {len(chunks)} chunks")
            await self._update_document_status(document_id, "processing", f"Generating embeddings ({len(chunks)} chunks)...")
            
            # Batching: Smaller batches (25) + Higher Concurrency (10) for speed
            batch_size = 25
            
            # Prepare batches with start index
            batches = []
            for i in range(0, len(chunks), batch_size):
                batch_data = chunks[i:i + batch_size]
                batches.append((i, batch_data))
            
            total_batches = len(batches)
            
            # Concurrency control (Limit to 10 to simulate ~10 concurrent users/requests)
            semaphore = asyncio.Semaphore(10)
            
            async def process_batch(batch_idx, start_index, batch_data):
                async with semaphore:
                    retries = 3
                    for attempt in range(retries):
                        try:
                            logger.info(f"Processing batch {batch_idx + 1}/{total_batches}")
                            
                            # 1. Embed
                            batch_embeddings = await embedding_service.generate_embeddings(batch_data)
                            
                            # 2. Metadata
                            batch_metadata = [
                                {
                                    "document_id": document_id,
                                    "document_name": filename,
                                    "chunk_id": start_index + k
                                }
                                for k in range(len(batch_data))
                            ]
                            
                            # 3. Upsert
                            await qdrant_service.upsert_chunks(
                                collection_name=collection_name,
                                chunks=batch_data,
                                embeddings=batch_embeddings,
                                metadata=batch_metadata
                            )
                            return
                            
                        except Exception as e:
                            if "429" in str(e) or "Too Many Requests" in str(e):
                                if attempt < retries - 1:
                                    wait_time = (2 ** attempt) + (0.1 * (batch_idx % 5)) # Jitter
                                    logger.warning(f"Rate limit for batch {batch_idx + 1}, retrying in {wait_time}s...")
                                    await asyncio.sleep(wait_time)
                                    continue
                            logger.error(f"Error in batch {batch_idx + 1} (Attempt {attempt+1}): {e}")
                            if attempt == retries - 1:
                                raise e

            # Create and run tasks
            tasks = [process_batch(idx, start_idx, b) for idx, (start_idx, b) in enumerate(batches)]
            await asyncio.gather(*tasks)
            
            # 6. Update status to completed
            await self._update_document_status(document_id, "completed")
            logger.info(f"Document {filename} processed successfully")
            
            # 7. Generate Topics
            try:
                from services.mcq_service import mcq_service
                await mcq_service.generate_document_topics(project_id, document_id)
            except Exception as topic_err:
                logger.error(f"Failed to generate topics for {filename}: {topic_err}")
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            await self._update_document_status(document_id, "failed", str(e))
    
    async def _update_document_status(
        self,
        document_id: str,
        status: str,
        message: Optional[str] = None
    ):
        """Update document processing status in database"""
        try:
            update_data = {"upload_status": status}
            if status == "completed":
                 update_data["error_message"] = None
            elif message:
                update_data["error_message"] = message
            
            self.client.table("documents").update(update_data).eq(
                "id", document_id
            ).execute()
            
        except Exception as e:
            logger.error(f"Error updating document status: {str(e)}")

    async def delete_document(self, project_id: str, document_id: str):
        """Delete document from DB and Vector Store"""
        try:
            # 1. Delete from Qdrant
            collection_name = f"project_{project_id}"
            await qdrant_service.delete_vectors(collection_name, document_id)
            
            # 2. Delete from DB
            # We need to know project_id. The caller passes it or we fetch it.
            # If we didn't have project_id, we'd query it first.
            
            self.client.table("documents").delete().eq("id", document_id).execute()
            
            logger.info(f"Deleted document {document_id} from project {project_id}")
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise

document_service = DocumentService()
