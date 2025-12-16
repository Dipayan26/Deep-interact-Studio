

#####################################


from fastapi import FastAPI, File, UploadFile
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
    """
    Receives a CSV file, processes it, and returns results
    """

    # 1. Read file content (bytes)
    content = await file.read()

    # 2. Convert bytes → pandas DataFrame
    df = pd.read_csv(io.BytesIO(content))

    # 3. Example processing
    num_rows = len(df)
    num_columns = len(df.columns)

    # Example: if column named 'value' exists
    mean_value = None
    if "value" in df.columns:
        mean_value = float(df["value"].mean())

    # 4. Return result as JSON
    return {
        "filename": file.filename,
        "rows": num_rows,
        "columns": num_columns,
        "mean_value": mean_value
    }