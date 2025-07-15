"""Web API for headline sentiment analysis service"""

import logging
from pathlib import Path
from typing import List, Tuple, Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

## Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Headline Sentiment Analysis API")

class HeadlinesRequest(BaseModel):
    """Request payload containing the list of news headlines to be scored."""

    headlines: List[str]


class HeadlinesResponse(BaseModel):
    """Response payload containing the list of sentiment labels."""

    labels: List[str]


def _load_sentence_model() -> SentenceTransformer:
    """Load a SentenceTransformer model"""

    local_model_path = Path("/opt/huggingface_models/all-MiniLM-L6-v2")
    if local_model_path.exists():
        logger.info("Loading sentence model from local path: %s", local_model_path)
        return SentenceTransformer(str(local_model_path))

    logger.info("Loading sentence model from Hugging Face hub")
    return SentenceTransformer("all-MiniLM-L6-v2")


def _load_svm_model() -> Any:
    """Load the pre‑trained SVM classifier"""

    svm_path = Path(__file__).parent / "svm.joblib"
    if not svm_path.exists():
        error_msg = f"SVM model not found at {svm_path}"
        logger.error("%s", error_msg)
        raise FileNotFoundError(error_msg)

    logger.info("Loading SVM model from: %s", svm_path)
    return joblib.load(svm_path)

def _load_models() -> Tuple[SentenceTransformer, Any]:
    """Load and return all models required by the API."""

    try:
        sentence_model_local = _load_sentence_model()
        svm_model_local = _load_svm_model()
        logger.info("Models loaded successfully")
        return sentence_model_local, svm_model_local
    except Exception as exc:
        logger.critical("Failed to load models: %s", exc)
        raise


SENTENCE_MODEL, SVM_MODEL = _load_models()


@app.get("/status", response_model=dict[str, str])
async def status() -> dict[str, str]:
    """Health‑check endpoint."""

    logger.debug("Status endpoint hit")
    return {"status": "OK"}


@app.post("/score_headlines", response_model=HeadlinesResponse)
async def score_headlines(request: HeadlinesRequest) -> HeadlinesResponse:
    """Score a list of headlines and return their sentiment labels."""

    if not request.headlines:
        logger.warning("Empty headlines list received")
        raise HTTPException(status_code=400, detail="Headlines list cannot be empty")

    try:
        # Generate embeddings and predict
        embeddings = SENTENCE_MODEL.encode(request.headlines, convert_to_numpy=True)
        labels = SVM_MODEL.predict(embeddings).tolist()
        logger.info("Successfully scored %s headlines", len(labels))
        return HeadlinesResponse(labels=labels)

    except HTTPException:
        raise  # Re‑raise FastAPI exceptions unchanged

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Error processing headlines: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing headlines: {exc}",
        ) from exc