from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from app.models.ai_model import AIModel
from app.schemas.ai_model import AIModelCreate
from sqlalchemy import or_
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIModelRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, model_data: AIModelCreate) -> AIModel:
        logger.info(f"Creating new model with name: {model_data.name}")
        db_model = AIModel(
            name=model_data.name,
            description=model_data.description,
            prompt=model_data.prompt,
            model_type=model_data.model_type,
            status="pending",
            config={}  # Initialize empty config
        )
        try:
            self.db.add(db_model)
            self.db.commit()
            self.db.refresh(db_model)
            logger.info(f"Successfully created model with ID: {db_model.id}")
            return db_model
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            self.db.rollback()
            raise

    def get_by_id(self, model_id: int) -> Optional[AIModel]:
        logger.info(f"Fetching model with ID: {model_id}")
        model = self.db.query(AIModel).filter(AIModel.id == model_id).first()
        if model:
            logger.info(f"Found model: {model.name}")
        else:
            logger.warning(f"Model with ID {model_id} not found")
        return model

    def update_status(
        self,
        model_id: int,
        status: str,
        config: Dict = None,
        mlflow_run_id: str = None
    ):
        model = self.get_by_id(model_id)
        if model:
            model.status = status
            if config:
                model.config = config
            if mlflow_run_id:
                model.mlflow_run_id = mlflow_run_id
            self.db.commit()
            self.db.refresh(model)
        return model

    def get_active_trainings(self) -> List[AIModel]:
        logger.info("Fetching active trainings")
        models = self.db.query(AIModel)\
            .filter(AIModel.status.in_(["pending", "training"]))\
            .all()
        logger.info(f"Found {len(models)} active trainings")
        return models

    def get_all(self, skip: int = 0, limit: int = 10) -> List[AIModel]:
        logger.info(f"Fetching all models with skip={skip}, limit={limit}")
        models = self.db.query(AIModel)\
            .order_by(AIModel.created_at.desc())\
            .offset(skip)\
            .limit(limit)\
            .all()
        logger.info(f"Found {len(models)} models")
        return models

    def count_all(self) -> int:
        return self.db.query(AIModel).count()

    def count_by_status(self, status: str) -> int:
        return self.db.query(AIModel).filter(AIModel.status == status).count()

    def get_completed_models(self) -> List[AIModel]:
        """Get all models with status 'completed'"""
        return self.db.query(AIModel)\
            .filter(AIModel.status == "completed")\
            .order_by(AIModel.updated_at.desc())\
            .all() 