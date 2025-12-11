from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class SumInput(BaseModel):
    a: float
    b: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sum")
def sum_numbers(data: SumInput):
    result = data.a + data.b
    return {"result": result}



