from fastapi import FastAPI

app = FastAPI(title="grounded")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
