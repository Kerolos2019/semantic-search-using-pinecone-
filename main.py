## Import Libraries
from fastapi import FastAPI, Form, HTTPException
from utils import search_vectDB



search = FastAPI(debug=True)

@search.post('/semantic_search')
async def semantic_search(search_text: str=Form(...), top_k: int=Form(100), 
                          threshold: float=Form(None), class_type: str=Form(..., description='class_type', enum=['All', 'class-a', 'class-b'])):

   
    if top_k <= 0 or not isinstance(top_k, int) or top_k > 10000 or top_k is None:
        raise HTTPException(status_code=400, detail="Bad Request: 'top_k' must be a positive integer and less than 10000.")
    
    elif threshold is not None and (threshold <= 0.0 or not isinstance(threshold, float) or threshold > 1.0):
        raise HTTPException(status_code=400, 
                            detail="Bad Request: 'threshold' must be a positive float greater than 0.0 and less than 1.0")

    else:
        
        similar_records = search_vectDB(query_text=search_text, top_k=top_k, threshold=threshold, class_type=class_type)

        return similar_records
    