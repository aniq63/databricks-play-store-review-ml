import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Fix for threading/GUI issues in FastAPI
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from src.logger import logging
from src.exception import MyException


class InsightGenerator:

    def __init__(self, output_dir: str = "outputs/insights"):
        try:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
            sns.set(style="whitegrid")
            logging.info(f"InsightGenerator initialized. Outputs will be saved to: {self.output_dir}")
        except Exception as e:
            logging.error(f"Error initializing InsightGenerator: {e}")
            raise MyException(e, sys)

    # --------------------------------------------------
    # Sentiment Distribution
    # --------------------------------------------------

    def sentiment_distribution(self, df: pd.DataFrame):
        try:
            logging.info("Generating sentiment distribution plot...")
            plt.figure(figsize=(8, 5))

            sns.countplot(
                data=df,
                x="sentiment",
                order=["Negative", "Neutral", "Positive"]
            )

            plt.title("Sentiment Distribution of Reviews")
            plt.xlabel("Sentiment")
            plt.ylabel("Number of Reviews")

            path = os.path.join(self.output_dir, "sentiment_distribution.png")
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            
            logging.info(f"Saved sentiment plot to {path}")
            return path
        except Exception as e:
            raise MyException(e, sys)

    # --------------------------------------------------
    # Topic Frequency
    # --------------------------------------------------

    def topic_frequency(self, df: pd.DataFrame):
        try:
            logging.info("Generating top topics frequency plot...")
            topic_counts = (
                df["clean_topic"]
                .value_counts()
                .head(10)
            )

            plt.figure(figsize=(10, 6))

            sns.barplot(
                x=topic_counts.values,
                y=topic_counts.index
            )

            plt.title("Top 10 Review Topics")
            plt.xlabel("Number of Reviews")
            plt.ylabel("Topic")

            path = os.path.join(self.output_dir, "top_topics.png")
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            
            logging.info(f"Saved topics plot to {path}")
            return path
        except Exception as e:
            raise MyException(e, sys)

    # --------------------------------------------------
    # Sentiment per Topic
    # --------------------------------------------------

    def topic_sentiment_heatmap(self, df: pd.DataFrame):
        try:
            logging.info("Generating topic sentiment heatmap...")
            pivot = pd.pivot_table(
                df,
                index="clean_topic",
                columns="sentiment",
                aggfunc="size",
                fill_value=0
            )

            plt.figure(figsize=(12, 7))

            sns.heatmap(
                pivot,
                annot=True,
                fmt="d",
                cmap="coolwarm"
            )

            plt.title("Sentiment Distribution per Topic")

            path = os.path.join(self.output_dir, "topic_sentiment_heatmap.png")
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            
            logging.info(f"Saved heatmap to {path}")
            return path
        except Exception as e:
            raise MyException(e, sys)

    # --------------------------------------------------
    # Top Negative Topics
    # --------------------------------------------------

    def top_negative_topics(self, df: pd.DataFrame):
        try:
            logging.info("Extracting top negative topics...")
            negative_df = df[df["sentiment"] == "Negative"]

            result = (
                negative_df["clean_topic"]
                .value_counts()
                .head(5)
                .reset_index()
            )

            result.columns = ["topic", "negative_reviews"]
            return result
        except Exception as e:
            raise MyException(e, sys)

    # --------------------------------------------------
    # Main Insight Pipeline
    # --------------------------------------------------

    def generate_all(self, df: pd.DataFrame):
        try:
            logging.info("Starting complete insight generation pipeline...")
            outputs = {}

            outputs["sentiment_plot"] = self.sentiment_distribution(df)
            outputs["topics_plot"] = self.topic_frequency(df)
            outputs["heatmap"] = self.topic_sentiment_heatmap(df)
            outputs["top_negative_topics"] = self.top_negative_topics(df)

            logging.info("Complete insight pipeline finished.")
            return outputs
        except Exception as e:
            logging.error(f"Error generating insights: {e}")
            raise MyException(e, sys)