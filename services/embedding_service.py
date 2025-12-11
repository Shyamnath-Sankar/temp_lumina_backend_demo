from langchain_together import TogetherEmbeddings
from config.settings import settings
from typing import List
from utils.logger import logger
import os
import asyncio

class EmbeddingService:
    def __init__(self):
        os.environ['TOGETHER_API_KEY'] = settings.TOGETHER_API_KEY
        self.embeddings = TogetherEmbeddings(
            model=settings.EMBEDDING_MODEL,
            together_api_key=settings.TOGETHER_API_KEY
        )
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            # LangChain handles batching internally usually, but explicit batching is safer
            embeddings = []
            loop = asyncio.get_running_loop()
            batch_size = 25
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = await loop.run_in_executor(
                    None,
                    lambda: self.embeddings.embed_documents(batch)
                )
                embeddings.extend(batch_embeddings)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

embedding_service = EmbeddingService()