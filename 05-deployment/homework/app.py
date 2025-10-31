import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Define the data model for the input
class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Load the pipeline
with open('pipeline_v1.bin', 'rb') as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(client: Client):
    data = client.dict()
    prob = model.predict_proba([data])[0, 1]
    return {"subscription_probability": round(float(prob), 3)}
