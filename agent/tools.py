from ddgs import DDGS
from typing import List, Dict

def duckduckgo_search(query: str, max_results: int = 10) -> List[Dict]:
    """
    Search DuckDuckGo and return results
    """
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                query=query,
                region='wt-wt',    
                safesearch='moderate',
                timelimit=None,     
                max_results=max_results
            )
            full_context = "".join([f"{result['title']} {result['body']}" for result in results])
            return full_context
            
    except Exception as e:
        print(f"Search error: {e}")
        return []




# if __name__ == "__main__":
#     query = "best small LLM models 2026"
    
#     results = duckduckgo_search(query, max_results=3)
    
#     print("Search Results:")
#     print(results)