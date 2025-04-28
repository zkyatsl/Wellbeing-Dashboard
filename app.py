from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model ElasticNet yang sudah dilatih
elastic_model = joblib.load('elasticnet_model.pkl')

# Inisialisasi aplikasi FastAPI
app = FastAPI()

# Definisikan skema input data untuk validasi
class InputData(BaseModel):
    TODO_COMPLETED: float
    SUFFICIENT_INCOME: float
    DAILY_STRESS: float
    FRUITS_VEGGIES: float
    ACHIEVEMENT: float

# Endpoint untuk prediksi dengan ElasticNet
@app.post("/predict_elasticnet/")
async def predict_elasticnet(data: InputData):
    # Mengubah input data menjadi DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Scaling data sebelum diprediksi (karena ElasticNet sensitif terhadap skala fitur)
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    
    # Prediksi dengan model ElasticNet
    prediction = elastic_model.predict(input_data_scaled)
    
    # Kembalikan hasil prediksi dalam format JSON
    return {"prediction_elasticnet": prediction[0]}