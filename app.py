from fastapi import FastAPI, Request
import mlflow.pyfunc

app = FastAPI()

RUN_ID = '2648c5c0cea24050b86fcc6adaa0b565'
model = mlflow.pyfunc.load_model(f'mlruns/721578769661279971/{RUN_ID}/artifacts/model')

@app.post("/predict/")
async def predict(request: Request):
    data = await request.json()
    X = [[data["feature"]]]
    prediction = model.predict(X)[0]
    return {"prediction": prediction}