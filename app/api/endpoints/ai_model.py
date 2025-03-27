from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Body
from sqlalchemy.orm import Session
from typing import List
from app.api import deps
from app.schemas.ai_model import AIModelCreate, AIModelResponse
from app.repositories.ai_model import AIModelRepository
from app.services.ai_generator import AIGenerator
from app.services.model_trainer import ModelTrainer
import mlflow
from app.core.config import settings
import logging
from fastapi import BackgroundTasks
from datetime import datetime
from pydantic import BaseModel
import numpy as np
import os
import joblib
from fastapi.responses import FileResponse
import zipfile
import tempfile
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])

# Tambahkan model Pydantic untuk request
class PredictionRequest(BaseModel):
    texts: List[str]

@router.post("/", response_model=AIModelResponse)
async def create_ai_model(
    background_tasks: BackgroundTasks,
    db: Session = Depends(deps.get_db),
    name: str = Form(...),
    description: str = Form(None),
    prompt: str = Form(...),
    dataset: UploadFile = File(...)
):
    try:
        # Simpan dataset
        dataset_path = await deps.save_upload_file(dataset)
        
        # Analisis prompt
        ai_generator = AIGenerator()
        model_config = ai_generator.analyze_prompt(prompt)
        
        # Buat record model
        repo = AIModelRepository(db)
        model_data = AIModelCreate(
            name=name,
            description=description,
            prompt=prompt,
            model_type=model_config["ai_type"]
        )
        
        # Simpan ke database
        db_model = repo.create(model_data)
        
        # Start training process
        trainer = ModelTrainer(repo)
        background_tasks.add_task(
            trainer.start_training,
            db_model.id,
            dataset_path,
            model_config
        )
        
        return db_model
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models/{model_id}", response_model=AIModelResponse)
async def get_model_status(
    model_id: int,
    db: Session = Depends(deps.get_db)
):
    repo = AIModelRepository(db)
    model = repo.get_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model

@router.get("/models/{model_id}/metrics")
async def get_model_metrics(
    model_id: int,
    db: Session = Depends(deps.get_db)
):
    repo = AIModelRepository(db)
    model = repo.get_by_id(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    if model.status != "completed":
        return {
            "status": model.status,
            "message": "Model masih dalam proses training"
        }
    
    # Ambil metrics dari MLflow jika ada
    if model.mlflow_run_id:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        run = mlflow.get_run(model.mlflow_run_id)
        return {
            "status": "completed",
            "metrics": run.data.metrics
        }
    
    return {
        "status": model.status,
        "metrics": {}
    }

@router.post("/test-analysis/")
async def test_analysis(
    prompt: str = Body(..., example="Buat model untuk klasifikasi sentiment")
):
    try:
        ai_generator = AIGenerator()
        result = ai_generator.analyze_prompt(prompt)
        return {
            "status": "success",
            "config": result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/active")
async def get_active_trainings(
    db: Session = Depends(deps.get_db)
):
    repo = AIModelRepository(db)
    active_models = repo.get_active_trainings()
    
    return {
        "active_count": len(active_models),
        "models": [
            {
                "id": model.id,
                "name": model.name,
                "status": model.status,
                "started_at": model.created_at
            }
            for model in active_models
        ]
    }

@router.get("/models/{model_id}/status", response_model=dict)
async def get_model_status(
    model_id: int,
    db: Session = Depends(deps.get_db)
):
    repo = AIModelRepository(db)
    model = repo.get_by_id(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model tidak ditemukan")
    
    status_info = {
        "id": model.id,
        "name": model.name,
        "status": model.status,
        "created_at": model.created_at,
        "updated_at": model.updated_at
    }
    
    if model.status == "training":
        # Add training progress info
        status_info["progress"] = {
            "stage": "training",
            "started_at": model.updated_at,
            "running_time_minutes": _calculate_running_time(model.updated_at)
        }
    elif model.status == "completed":
        # Add completion info
        status_info["results"] = {
            "mlflow_run_id": model.mlflow_run_id,
            "metrics": _get_mlflow_metrics(model.mlflow_run_id) if model.mlflow_run_id else None
        }
    
    return status_info

def _calculate_running_time(updated_at: datetime) -> int:
    if not updated_at:
        return 0
    now = datetime.now(updated_at.tzinfo)
    diff = now - updated_at
    return round(diff.total_seconds() / 60)  # Return minutes

def _get_mlflow_metrics(run_id: str) -> dict:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    run = mlflow.get_run(run_id)
    return run.data.metrics

@router.get("/{model_id}/logs")
async def get_model_logs(
    model_id: int,
    db: Session = Depends(deps.get_db)
):
    repo = AIModelRepository(db)
    model = repo.get_by_id(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model tidak ditemukan")
    
    logs = {
        "model_info": {
            "id": model.id,
            "name": model.name,
            "type": model.model_type,
            "status": model.status
        },
        "configuration": model.config,
        "timeline": [
            {
                "stage": "created",
                "timestamp": model.created_at,
                "detail": "Model dibuat"
            }
        ]
    }
    
    if model.updated_at:
        logs["timeline"].append({
            "stage": model.status,
            "timestamp": model.updated_at,
            "detail": _get_status_detail(model.status)
        })
    
    return logs 

@router.get("/monitoring/active-trainings", response_model=dict)
async def get_active_trainings(
    db: Session = Depends(deps.get_db)
):
    """Get all active training models"""
    logger.info("Fetching active trainings")
    
    try:
        repo = AIModelRepository(db)
        active_models = repo.get_active_trainings()
        
        return {
            "active_count": len(active_models),
            "models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "status": model.status,
                    "started_at": model.created_at
                }
                for model in active_models
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching active trainings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/monitoring/summary")
async def get_training_summary(
    db: Session = Depends(deps.get_db)
):
    repo = AIModelRepository(db)
    return {
        "total_models": repo.count_all(),
        "active_trainings": repo.count_by_status("training"),
        "completed_models": repo.count_by_status("completed"),
        "failed_models": repo.count_by_status("failed"),
        "pending_models": repo.count_by_status("pending")
    }

@router.get("/list", response_model=dict)
async def list_models(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(deps.get_db)
):
    """Get list of all models"""
    logger.info(f"Fetching models with skip={skip}, limit={limit}")
    
    try:
        repo = AIModelRepository(db)
        models = repo.get_all(skip=skip, limit=limit)
        
        return {
            "total": len(models),
            "models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "status": model.status,
                    "created_at": model.created_at,
                    "updated_at": model.updated_at,
                    "model_type": model.model_type,
                    "config": model.config
                }
                for model in models
            ]
        }
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-create", response_model=dict)
async def test_create_model(
    db: Session = Depends(deps.get_db)
):
    """Test endpoint to create a sample model"""
    logger.info("Creating test model...")
    
    try:
        model_data = AIModelCreate(
            name="Test Sentiment Model",
            description="Test model for sentiment analysis",
            prompt="Create a model for sentiment analysis",
            model_type="classification"
        )
        
        repo = AIModelRepository(db)
        model = repo.create(model_data)
        
        logger.info(f"Test model created with ID: {model.id}")
        
        return {
            "message": "Test model created successfully",
            "model": {
                "id": model.id,
                "name": model.name,
                "status": model.status,
                "created_at": model.created_at
            }
        }
    except Exception as e:
        logger.error(f"Error creating test model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/completed")
async def get_completed_models(
    db: Session = Depends(deps.get_db)
):
    """Get all completed models with their metrics"""
    repo = AIModelRepository(db)
    completed_models = repo.get_completed_models()
    
    result = []
    for model in completed_models:
        model_info = {
            "id": model.id,
            "name": model.name,
            "created_at": model.created_at,
            "completed_at": model.updated_at,
            "model_type": model.model_type,
            "config": model.config
        }
        
        # Tambahkan metrics dari MLflow jika ada
        if model.mlflow_run_id:
            try:
                mlflow.set_tracking_uri("http://localhost:5001")
                run = mlflow.get_run(model.mlflow_run_id)
                model_info["metrics"] = run.data.metrics
            except Exception as e:
                model_info["metrics_error"] = str(e)
        
        result.append(model_info)
    
    return {
        "total_completed": len(completed_models),
        "models": result
    }

@router.get("/{model_id}/details")
async def get_model_details(
    model_id: int,
    db: Session = Depends(deps.get_db)
):
    """Get detailed information about a specific model"""
    repo = AIModelRepository(db)
    model = repo.get_by_id(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model tidak ditemukan")
    
    result = {
        "id": model.id,
        "name": model.name,
        "description": model.description,
        "prompt": model.prompt,
        "model_type": model.model_type,
        "status": model.status,
        "created_at": model.created_at,
        "updated_at": model.updated_at,
        "config": model.config
    }
    
    if model.status == "completed" and model.mlflow_run_id:
        try:
            mlflow.set_tracking_uri("http://localhost:5001")
            run = mlflow.get_run(model.mlflow_run_id)
            result["training_results"] = {
                "metrics": run.data.metrics,
                "parameters": run.data.params
            }
        except Exception as e:
            result["metrics_error"] = str(e)
    
    return result 

@router.post("/{model_id}/predict")
async def predict_sentiment(
    model_id: int,
    request: PredictionRequest,
    db: Session = Depends(deps.get_db)
):
    """Gunakan model untuk prediksi"""
    logger.info(f"Predicting using model {model_id}")
    
    repo = AIModelRepository(db)
    model = repo.get_by_id(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model tidak ditemukan")
        
    if model.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Model belum siap digunakan (status: {model.status})"
        )
    
    try:
        # Load model dan vectorizer dari path lokal
        model_path = model.config.get("model_path")
        if not model_path:
            raise ValueError("Model path not found in config")
            
        logger.info(f"Loading model from {model_path}")
        loaded_model = joblib.load(os.path.join(model_path, "model.joblib"))
        vectorizer = joblib.load(os.path.join(model_path, "vectorizer.joblib"))
        
        # Vectorize texts
        logger.info("Vectorizing input texts...")
        X = vectorizer.transform(request.texts)
        
        # Get predictions and probabilities
        logger.info("Making predictions...")
        predictions = loaded_model.predict(X)
        probabilities = loaded_model.predict_proba(X)
        
        # Format results
        results = []
        for text, pred, prob in zip(request.texts, predictions, probabilities):
            results.append({
                "text": text,
                "prediction": str(pred),
                "confidence": float(np.max(prob)),
                "probabilities": {
                    "positive": float(prob[1]) if len(prob) > 1 else 0.0,
                    "negative": float(prob[0])
                }
            })
        
        return {
            "model_name": model.name,
            "model_id": model.id,
            "predictions": results
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error saat melakukan prediksi: {str(e)}"
        )

# Tambahkan class untuk preprocessing
class TextPreprocessor:
    def __init__(self, steps):
        self.steps = steps
        
    def process(self, text: str) -> str:

        processed = text.lower()
        return processed 

@router.get("/{model_id}/download")
async def download_model(
    model_id: int,
    db: Session = Depends(deps.get_db)
):
    """Download trained model dan vectorizer"""
    repo = AIModelRepository(db)
    model = repo.get_by_id(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model tidak ditemukan")
        
    if model.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail="Model belum selesai training"
        )
    
    try:
        model_path = model.config.get("model_path")
        if not model_path:
            raise ValueError("Model path not found in config")
        
        # Buat temporary zip file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
            zip_path = tmp_zip.name
            
            # Create zip file
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add model files
                for root, _, files in os.walk(model_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, model_path)
                        zipf.write(file_path, arcname)
                
                # Add readme
                readme_content = f"""
                Model: {model.name}
                ID: {model.id}
                Type: {model.model_type}
                Created: {model.created_at}
                
                Files:
                - model.joblib: The trained model
                - vectorizer.joblib: The fitted vectorizer
                
                How to use:
                ```python
                import joblib
                
                # Load model and vectorizer
                model = joblib.load('model.joblib')
                vectorizer = joblib.load('vectorizer.joblib')
                
                # Predict
                text = "Your text here"
                X = vectorizer.transform([text])
                prediction = model.predict(X)
                probabilities = model.predict_proba(X)
                ```
                """
                
                zipf.writestr('README.txt', readme_content)
        
        return FileResponse(
            zip_path,
            media_type='application/zip',
            filename=f'model_{model_id}_{model.name.lower().replace(" ", "_")}.zip'
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saat mendownload model: {str(e)}"
        )
    finally:
        # Cleanup temporary file
        if 'zip_path' in locals():
            try:
                os.unlink(zip_path)
            except:
                pass

@router.get("/{model_id}/sample-code")
async def get_sample_code(
    model_id: int,
    db: Session = Depends(deps.get_db)
):
    """Get sample code untuk menggunakan model"""
    repo = AIModelRepository(db)
    model = repo.get_by_id(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model tidak ditemukan")
    
    sample_code = f"""
    # Cara menggunakan model {model.name} (ID: {model_id})
    
    import joblib
    
    def load_model():
        # Load model dan vectorizer
        model = joblib.load('model.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
        return model, vectorizer
    
    def predict_sentiment(text: str):
        # Load model
        model, vectorizer = load_model()
        
        # Vectorize text
        X = vectorizer.transform([text])
        
        # Get prediction dan probabilities
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        return {{
            "text": text,
            "prediction": prediction,
            "confidence": float(max(probabilities)),
            "probabilities": {{
                "positive": float(probabilities[1]),
                "negative": float(probabilities[0])
            }}
        }}
    
    # Contoh penggunaan
    if __name__ == "__main__":
        text = "Pelayanan sangat memuaskan"
        result = predict_sentiment(text)
        print(result)
    """
    
    return {"sample_code": sample_code} 