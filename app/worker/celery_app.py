from celery import Celery
from app.core.config import settings
import mlflow
from app.services.model_trainer import ModelTrainer

celery_app = Celery(
    "worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

@celery_app.task
def train_model_task(model_id: int, dataset_path: str, config: dict):
    try:
        # Setup MLflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        
        with mlflow.start_run() as run:
            # Train model
            trainer = ModelTrainer()
            metrics = trainer.train(dataset_path, config)
            
            # Log metrics dan parameters
            mlflow.log_params(config)
            mlflow.log_metrics(metrics)
            
            # Update model status
            from app.api.deps import get_db
            db = next(get_db())
            repo = AIModelRepository(db)
            repo.update_status(
                model_id=model_id,
                status="completed",
                mlflow_run_id=run.info.run_id
            )
            
            return {
                "status": "success",
                "mlflow_run_id": run.info.run_id,
                "metrics": metrics
            }
            
    except Exception as e:
        # Update status jika gagal
        repo.update_status(model_id=model_id, status="failed")
        raise e 