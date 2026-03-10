import os
import sys
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from keybert import KeyBERT

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.logger import logging
from src.exception import MyException


class ReviewClusterer:

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant"
    ):
        try:
            logging.info(f"Loading embedding model '{embedding_model}'...")
            self.embedder = SentenceTransformer(embedding_model)

            logging.info("Loading keyword extractor...")
            self.kw_model = KeyBERT(embedding_model)

            logging.info(f"Initializing LLM '{llm_model}'...")
            
            if not os.getenv("GROQ_API_KEY"):
                logging.warning("GROQ_API_KEY not found in environment variables. Topic cleaning may fail.")

            self.llm = ChatGroq(model_name=llm_model)

            prompt_text = """
            Analyze the following keywords and generate a short topic name (1–5 words).
            Return ONLY the topic name.

            Keywords: {words}
            """

            prompt = PromptTemplate.from_template(prompt_text)

            self.chain = prompt | self.llm | StrOutputParser()
        except Exception as e:
            logging.error(f"Error initializing ReviewClusterer: {e}")
            raise MyException(e, sys)

    # ---------------------------------------------------
    # STEP 1 — Embedding
    # ---------------------------------------------------

    def generate_embeddings(self, reviews):
        try:
            logging.info(f"Generating embeddings for {len(reviews)} reviews...")
            embeddings = self.embedder.encode(
                reviews,
                batch_size=64,
                show_progress_bar=True
            )
            return embeddings
        except Exception as e:
            raise MyException(e, sys)

    # ---------------------------------------------------
    # STEP 2 — UMAP Dimensionality Reduction
    # ---------------------------------------------------

    def reduce_dimensions(self, embeddings):
        try:
            logging.info(f"Reducing dimensions with UMAP for {len(embeddings)} embeddings...")
            reducer = umap.UMAP(
                n_components=10,
                n_neighbors=15,
                min_dist=0.0,
                metric="cosine",
                random_state=42
            )

            reduced_embeddings = reducer.fit_transform(embeddings)
            return reduced_embeddings
        except Exception as e:
            raise MyException(e, sys)

    # ---------------------------------------------------
    # STEP 3 — HDBSCAN Clustering
    # ---------------------------------------------------

    def cluster_embeddings(self, reduced_embeddings):
        try:
            logging.info("Clustering embeddings with HDBSCAN...")
            # Dynamically adjust min_cluster_size based on input size
            # For small datasets, we need a smaller cluster size
            n_samples = len(reduced_embeddings)
            dynamic_min_size = max(2, min(5, n_samples // 4)) if n_samples < 100 else 50
            dynamic_min_samples = max(1, dynamic_min_size // 2)

            logging.info(f"Clustering with min_cluster_size={dynamic_min_size}")
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=dynamic_min_size,
                min_samples=dynamic_min_samples,
                metric="euclidean",
                cluster_selection_method="eom"
            )

            labels = clusterer.fit_predict(reduced_embeddings)
            logging.info(f"Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")
            return labels
        except Exception as e:
            raise MyException(e, sys)

    # ---------------------------------------------------
    # STEP 4 — Keyword Extraction
    # ---------------------------------------------------

    def extract_topics(self, df):
        try:
            logging.info("Extracting keywords for each cluster using KeyBERT...")
            cluster_names = {}

            for cluster_id in sorted(df["cluster"].unique()):
                if cluster_id == -1:
                    cluster_names[-1] = "Uncategorized"
                    continue

                cluster_reviews = df[df["cluster"] == cluster_id]["content"].tolist()
                combined_text = " ".join(cluster_reviews[:200])

                keywords = self.kw_model.extract_keywords(
                    combined_text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words="english",
                    top_n=3,
                    use_mmr=True,
                    diversity=0.5
                )

                topic = " | ".join([k[0] for k in keywords])
                cluster_names[cluster_id] = topic

            df["topic"] = df["cluster"].map(cluster_names)
            return df
        except Exception as e:
            raise MyException(e, sys)

    # ---------------------------------------------------
    # STEP 5 — LLM Topic Cleaning
    # ---------------------------------------------------

    def clean_topics(self, df):
        try:
            logging.info("Cleaning topic names using LLM...")
            topics = df["topic"].unique()
            clean_topics = {}

            for topic in topics:
                if topic == "Uncategorized":
                    clean_topics[topic] = topic
                    continue
                try:
                    name = self.chain.invoke({"words": topic})
                    clean_topics[topic] = name.strip()
                except Exception as llm_e:
                    logging.warning(f"LLM topic cleaning failed for '{topic}': {llm_e}")
                    clean_topics[topic] = "Unknown Topic"

            df["clean_topic"] = df["topic"].map(clean_topics)
            return df
        except Exception as e:
            raise MyException(e, sys)

    # ---------------------------------------------------
    # MAIN PIPELINE
    # ---------------------------------------------------

    def run(self, df: pd.DataFrame):
        try:
            logging.info("Starting Review Clustering Pipeline...")
            reviews = df["content"].tolist()

            embeddings = self.generate_embeddings(reviews)
            reduced = self.reduce_dimensions(embeddings)
            labels = self.cluster_embeddings(reduced)
            df["cluster"] = labels

            df = self.extract_topics(df)
            df = self.clean_topics(df)
            
            logging.info("Clustering pipeline complete.")
            return df
        except Exception as e:
            logging.error(f"Error running clustering pipeline: {e}")
            raise MyException(e, sys)