"""Web API for headline sentiment analysis service"""

import logging
from pathlib import Path
from typing import Dict, List

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Headline Sentiment Analysis API")
sentence_model = None
svm_model = None


class HeadlinesRequest(BaseModel):
    """Request model for headline scoring endpoint."""
    headlines: List[str]


class HeadlinesResponse(BaseModel):
    """Response model for headline scoring endpoint."""
    labels: List[str]


def load_models():
    """Load the sentence transformer and SVM models once at startup."""
    global sentence_model, svm_model
    
    logger.info("Loading models...")
    
    try:
        # sentence transformer
        local_model_path = Path("/opt/huggingface_models/all-MiniLM-L6-v2")
        if local_model_path.exists():
            logger.info(f"Loading sentence model from local path: {local_model_path}")
            sentence_model = SentenceTransformer(str(local_model_path))
        else:
            logger.info("Loading sentence model from HuggingFace hub")
            sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # SVM model
        svm_path = Path(__file__).parent / "svm.joblib"
        if not svm_path.exists():
            logger.error(f"SVM model not found at {svm_path}")
            raise FileNotFoundError(f"SVM model not found at {svm_path}")
        
        logger.info(f"Loading SVM model from: {svm_path}")
        svm_model = joblib.load(svm_path)
        
        logger.info("Models loaded successfully!")
        
    except Exception as e:
        logger.critical(f"Failed to load models: {e}")
        raise


load_models()


@app.get("/status", response_model=Dict[str, str])
async def status():
    """Verify service is running."""
    logger.info("Status check requested")
    return {"status": "OK"}


@app.post("/score_headlines", response_model=HeadlinesResponse)
async def score_headlines(request: HeadlinesRequest):
    """
    Score a list of headlines and return their sentiment labels.
    
    Args:
        request: HeadlinesRequest containing list of headlines
        
    Returns:
        HeadlinesResponse containing list of labels (Optimistic/Pessimistic/Neutral)
    """
    try:
        # Log request info
        logger.info(f"Received request to score {len(request.headlines)} headlines")
        
        if not request.headlines:
            logger.warning("Empty headlines list received")
            raise HTTPException(status_code=400, detail="Headlines list cannot be empty")
        
        # Check if models are loaded
        if sentence_model is None or svm_model is None:
            logger.error("Models not loaded properly")
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Generate embeddings
        logger.debug("Generating embeddings...")
        embeddings = sentence_model.encode(request.headlines, convert_to_numpy=True)
        
        # Make predictions
        logger.debug("Making predictions...")
        predictions = svm_model.predict(embeddings)
        
        # Convert predictions to list
        labels = predictions.tolist()
        
        logger.info(f"Successfully scored {len(labels)} headlines")
        
        return HeadlinesResponse(labels=labels)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing headlines: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing headlines: {str(e)}")