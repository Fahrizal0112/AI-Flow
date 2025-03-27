import mlflow
import pandas as pd
from typing import Dict, Any
import logging
from app.repositories.ai_model import AIModelRepository
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np
import os
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, repo: AIModelRepository):
        self.repo = repo
        # Buat absolute path untuk model directory
        self.model_dir = os.path.abspath(os.path.join(os.getcwd(), "saved_models"))
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"Model directory: {self.model_dir}")

    async def start_training(self, model_id: int, dataset_path: str, config: Dict[str, Any]):
        """Mulai proses training"""
        logger.info(f"Starting training for model {model_id}")
        
        try:
            # Update status ke training
            self.repo.update_status(model_id, "training")
            
            # Load dan preprocess dataset
            data = pd.read_csv(dataset_path)
            logger.info(f"Dataset loaded with shape: {data.shape}")
            
            # Setup MLflow
            mlflow.set_tracking_uri("http://localhost:5001")
            
            with mlflow.start_run() as run:
                # Preprocess
                vectorizer = TfidfVectorizer(max_features=5000)
                X = vectorizer.fit_transform(data['text'])
                y = data['sentiment']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train model
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                # Log metrics
                metrics = {
                    "train_accuracy": train_score,
                    "test_accuracy": test_score
                }
                mlflow.log_metrics(metrics)
                
                # Buat model directory yang unique
                model_path = os.path.join(self.model_dir, f"model_{model_id}")
                os.makedirs(model_path, exist_ok=True)
                
                # Save model dan vectorizer
                joblib.dump(model, os.path.join(model_path, "model.joblib"))
                joblib.dump(vectorizer, os.path.join(model_path, "vectorizer.joblib"))
                
                # Update config dengan absolute path
                config["model_path"] = model_path
                
                # Update model status
                self.repo.update_status(
                    model_id=model_id,
                    status="completed",
                    config=config,
                    mlflow_run_id=run.info.run_id
                )
                
                logger.info(f"Training completed for model {model_id}")
                return metrics
                
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            self.repo.update_status(model_id, "failed")
            raise

    async def _train_model(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, float]:
        """Actual training process"""
        try:
            # Simulate training process
            logger.info("Preprocessing data...")
            # Text vectorization
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(data['text'])
            y = data['sentiment']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Train model
            logger.info("Training model...")
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train, y_train)

            # Evaluate
            accuracy = model.score(X_test, y_test)
            
            return {
                "accuracy": accuracy,
                "training_time": time.time()
            }

        except Exception as e:
            logger.error(f"Error in training process: {str(e)}")
            raise

    def _train_classification(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, float]:
        """Train classification model"""
        # Implementasi training untuk classification
        # Ini adalah contoh sederhana, sesuaikan dengan kebutuhan
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score
        
        # Prepare data
        X = data[config["features"]]
        y = data[config["target"]]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train model
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics

    def _train_regression(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, float]:
        """Train regression model"""
        # Implementasi untuk regression
        # Tambahkan kode sesuai kebutuhan
        return {"mse": 0.0, "r2": 0.0}

    def _train_clustering(self, data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, float]:
        """Train clustering model"""
        # Implementasi untuk clustering
        # Tambahkan kode sesuai kebutuhan
        return {"silhouette_score": 0.0}
