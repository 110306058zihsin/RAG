from fastapi import FastAPI
import embed
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s : %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

app = FastAPI()

app.include_router(embed.router)
