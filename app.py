from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from utils import SemanticSearchEngine   

app = FastAPI()

search_engine = SemanticSearchEngine()

class SearchQuery(BaseModel):
    query: str

@app.post("/search_without_llm", response_model=List[dict])
async def search_without_llm(query: SearchQuery):
    """
    Recherche des livres similaires sans l'utilisation du LLM pour l'amélioration des recommandations.
    """
    try:
        results = search_engine.search_books_by_query(query.query)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/search_with_llm", response_model=str)
async def search_with_llm(query: SearchQuery):
    """
    Recherche des livres similaires avec l'amélioration des recommandations par un LLM.
    """
    try:
        results = search_engine.search_books_by_query(query.query)
        
        refined_results = search_engine.refine_recommendations_with_llm(query.query, results)
        return refined_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

