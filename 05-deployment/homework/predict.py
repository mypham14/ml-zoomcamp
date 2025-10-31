import pickle

# Load the pipeline
with open('pipeline_v1.bin', 'rb') as f_in:
    model = pickle.load(f_in)

# The record to score
record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Get probability of conversion (class 1)
prob = model.predict_proba([record])[0, 1]

print("Probability of conversion:", prob)