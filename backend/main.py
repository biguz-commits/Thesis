from dotenv import load_dotenv


# load the environment variables
load_dotenv()


import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routes import health
from app.routes import invoke



@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup...")
    try:
        yield
    finally:
        print("Application shutdown...")


prefix = "/thesis"

app = FastAPI(
    title="Recomandation System for Amazon Products",
    lifespan=lifespan,
)

app.include_router(health.router, prefix=prefix,  tags = ["health"])
app.include_router(invoke.router, prefix=prefix,  tags = ["invoke"])


def run_dev():
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False, log_level="info")

if __name__ == "__main__":
   uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False, log_level="info")