from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI

import journal_api_8
import logging


logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s : %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger(__name__)

app = FastAPI()

#app.include_router(multi_requests.router)
app.include_router(journal_api_8.router)