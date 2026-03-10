import os
import re
import string
import requests
import pandas as pd
from typing import List, Dict

import sys
from src.logger import logging
from src.exception import MyException


class PlayStoreSentimentAPI:

    def __init__(self, endpoint_url: str = None, token: str = None):
        try:
            self.token = token or os.getenv("DATABRICKS_TOKEN")
            if not self.token:
                raise ValueError("DATABRICKS_TOKEN is not set")

            if endpoint_url:
                self.endpoint_url = endpoint_url
            elif os.getenv("DATABRICKS_ENDPOINT"):
                self.endpoint_url = os.getenv("DATABRICKS_ENDPOINT")
            else:
                db_host = os.getenv("DATABRICKS_HOST", "")
                endpoint_name = os.getenv("DATABRICKS_ENDPOINT_NAME", "play_store_reviews_model_v1")
                
                if not db_host:
                    logging.warning("DATABRICKS_HOST is not set while trying to construct endpoint URL.")
                
                # Construct default Databricks Serving URL
                self.endpoint_url = f"{db_host.rstrip('/')}/serving-endpoints/{endpoint_name}/invocations"

            logging.info(f"Initialized PlayStoreSentimentAPI with endpoint: {self.endpoint_url}")
        except Exception as e:
            logging.error(f"Error initializing API Client: {e}")
            raise MyException(e, sys)

        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        # sentiment label mapping
        self.label_map = {
            0: "Negative",
            1: "Neutral",
            2: "Positive"
        }

    # -----------------------------
    # Text Preprocessing
    # -----------------------------
    def preprocess_text(self, text: str) -> str:
        try:
            text = text.lower()
            text = re.sub(r"http\S+", "", text)  # remove urls
            text = re.sub(r"\d+", "", text)  # remove numbers
            text = text.translate(str.maketrans("", "", string.punctuation))
            text = re.sub(r"\s+", " ", text).strip()
            return text
        except Exception as e:
            raise MyException(e, sys)

    def preprocess_reviews(self, reviews: List[str]) -> List[str]:
        try:
            return [self.preprocess_text(r) for r in reviews]
        except Exception as e:
            raise MyException(e, sys)

    # -----------------------------
    # Prediction
    # -----------------------------
    def predict(self, reviews: List[str]) -> List[Dict]:
        try:
            logging.info(f"Sending {len(reviews)} reviews for prediction...")
            
            # preprocessing
            processed_reviews = self.preprocess_reviews(reviews)

            df = pd.DataFrame({
                "content": processed_reviews
            })

            payload = {
                "dataframe_records": df.to_dict(orient="records")
            }

            response = requests.post(
                self.endpoint_url,
                headers=self.headers,
                json=payload
            )

            if response.status_code != 200:
                raise Exception(
                    f"Request failed: {response.status_code}, {response.text}"
                )

            result = response.json()["predictions"]

            # convert numeric labels to sentiment names
            sentiments = [self.label_map.get(p, "Unknown") for p in result]

            output = []

            for review, sentiment in zip(reviews, sentiments):
                output.append({
                    "review": review,
                    "sentiment": sentiment
                })

            logging.info("Predictions successfully retrieved.")
            return output

        except Exception as e:
            logging.error(f"Error during prediction request: {e}")
            raise MyException(e, sys)