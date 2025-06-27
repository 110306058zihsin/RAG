from fastapi import APIRouter
from pydantic import BaseModel
import logging
from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv

# load_dotenv()
log = logging.getLogger(__name__)

router = APIRouter()

embedding_model = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True)

class embedReq(BaseModel):
    input: str


class embedResp(BaseModel):
    embedding: list[float]


@router.post("/BAAI/bge-m3", response_model=embedResp)
async def embed_text(req: embedReq):
    input = req.input
    log.info(f'input:{input}')
    embedding = embedding_model.encode(input) 
    return {"embedding": embedding.tolist()}
