import os
import requests
from typing import List, Dict, Optional
import sys
from src.logger import logging
from src.exception import MyException

class SerpApiScraper:
    """
    Scraper for Google Play Store reviews using SerpApi.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            logging.error("SERPAPI_API_KEY not found in environment or arguments.")
            raise ValueError("SERPAPI_API_KEY is required for SerpApiScraper")
        
        self.base_url = "https://serpapi.com/search"
        logging.info("SerpApiScraper initialized.")

    def fetch_reviews(self, product_id: str, max_reviews: int = 100) -> List[str]:
        """
        Fetch reviews for a given Google Play product ID.
        
        Args:
            product_id: The package name of the app (e.g., 'com.google.android.apps.maps').
            max_reviews: Maximum number of reviews to fetch (multiples of 199 up to max).
            
        Returns:
            A list of review snippets (text).
        """
        all_review_texts = []
        next_page_token = None
        
        try:
            while len(all_review_texts) < max_reviews:
                params = {
                    "engine": "google_play_product",
                    "product_id": product_id,
                    "store": "apps",
                    "all_reviews": "true",
                    "api_key": self.api_key,
                    "num": min(199, max_reviews - len(all_review_texts))
                }
                
                if next_page_token:
                    params["next_page_token"] = next_page_token
                
                logging.info(f"Fetching reviews for {product_id}, page token: {next_page_token}")
                response = requests.get(self.base_url, params=params)
                
                if response.status_code != 200:
                    logging.error(f"SerpApi request failed with status {response.status_code}: {response.text}")
                    break
                    
                data = response.json()
                reviews = data.get("reviews", [])
                
                if not reviews:
                    logging.info("No more reviews found.")
                    break
                    
                for r in reviews:
                    snippet = r.get("snippet")
                    if snippet:
                        all_review_texts.append(snippet)
                        
                # Check for next page
                serpapi_pagination = data.get("serpapi_pagination")
                if serpapi_pagination and "next_page_token" in serpapi_pagination:
                    next_page_token = serpapi_pagination["next_page_token"]
                else:
                    logging.info("No more pages available.")
                    break
                    
                if len(all_review_texts) >= max_reviews:
                    logging.info(f"Reached max_reviews limit: {max_reviews}")
                    break
            
            logging.info(f"Successfully fetched {len(all_review_texts)} reviews for {product_id}")
            return all_review_texts[:max_reviews]
            
        except Exception as e:
            logging.error(f"Error fetching reviews from SerpApi: {e}")
            raise MyException(e, sys)
