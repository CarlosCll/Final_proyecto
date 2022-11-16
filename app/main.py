from app.predictor import Predictor
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
#import uvicorn
from starlette.responses import RedirectResponse
import os

class PredictionSchema(BaseModel):
    filename:str
    prediction: str
    top: dict
app = FastAPI(title="Clasificador de papa")

predictor=Predictor("clases.txt","New_Model.h5","clases_tipologia.txt","model_MOBILNETv2.h5",224)
@app.get("/")
def raiz():
   return RedirectResponse(url="/docs/")

@app.post("/predict", response_model=PredictionSchema)
def predict_image( opcion : int, file: UploadFile = File(...)):
    prediction = predictor.predict_file(file.file, opcion=opcion)

    return PredictionSchema(
        filename=file.filename,
        prediction=prediction["label"],
         top=prediction["top"]
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))