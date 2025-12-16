

#####################################


from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import io
app = FastAPI()


# class SumInput(BaseModel):
#     a: float
#     b: float


@app.get("/health")
def health():
    return {"status": "ok"}


# @app.post("/sum")
# def sum_numbers(data: SumInput):
#     result = data.a + data.b
#     return {"result": result}

@app.post("/process_csv")
async def process_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)   # directly read from file-like object
    return {
        "rows": len(df),
        "filename": file.filename
    }


