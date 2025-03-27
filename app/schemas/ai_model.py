from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class AIModelBase(BaseModel):
    name: str
    description: Optional[str] = None
    prompt: str
    model_type: Optional[str] = None

class AIModelCreate(AIModelBase):
    pass

class AIModelResponse(AIModelBase):
    id: int
    status: str
    config: Optional[Dict[str, Any]] = None
    mlflow_run_id: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True 