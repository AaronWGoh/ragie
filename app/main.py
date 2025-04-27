from fastapi import FastAPI
from api.endpoints import router

app = FastAPI(title="Ragie RAG System")

# Include the router
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
