from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. API Uygulamasını Başlatıyoruz
app = FastAPI(
    title="Bankacılık Müşteri Segmentasyon API",
    description="Müşteri finansal verilerini alıp, K-Means modeli ile segmentasyon yapar.",
    version="1.0.0"
)

# 2. Kaydettiğimiz Modeli ve Pipeline'ı RAM'e Yüklüyoruz
artifacts = joblib.load('customer_segmentation_model.pkl')
preprocessing_pipeline = artifacts['pipeline']
kmeans_model = artifacts['model']

# 3. Gelen İsteğin Veri Şemasını (Contract) Pydantic ile Tanımlıyoruz
class CustomerData(BaseModel):
    BALANCE: float
    BALANCE_FREQUENCY: float
    PURCHASES: float
    ONEOFF_PURCHASES: float
    INSTALLMENTS_PURCHASES: float
    CASH_ADVANCE: float
    PURCHASES_FREQUENCY: float
    ONEOFF_PURCHASES_FREQUENCY: float
    PURCHASES_INSTALLMENTS_FREQUENCY: float
    CASH_ADVANCE_FREQUENCY: float
    CASH_ADVANCE_TRX: int
    PURCHASES_TRX: int
    CREDIT_LIMIT: float
    PAYMENTS: float
    MINIMUM_PAYMENTS: float
    PRC_FULL_PAYMENT: float
    TENURE: int

# 4. Tahmin Yapan Endpoint'i (Uç Noktayı) Kodluyoruz
@app.post("/predict")
def predict_segment(data: CustomerData):
    # API'ye gelen JSON verisini pandas DataFrame'e dönüştür
    df_input = pd.DataFrame([data.dict()])
    
    # 17 boyutlu ham veriyi Pipeline'dan geçir (İmputasyon, Log1p, Scaler, PCA)
    processed_data = preprocessing_pipeline.transform(df_input)
    
    # Modeli çalıştır ve küme (Cluster) numarasını al
    cluster_id = kmeans_model.predict(processed_data)[0]
    
    # Kümeleri iş mantığındaki (Domain) personalarla eşleştir
    personas = {
        0: "Premium / Elit Harcamacılar",
        1: "İnaktif / Cüzdanda Unutanlar",
        2: "Bütçe Odaklı Taksit Severler",
        3: "Nakit Avans Bağımlıları / Borçlananlar"
    }
    
    return {
        "cluster_id": int(cluster_id),
        "persona": personas.get(int(cluster_id), "Bilinmeyen Profil"),
        "status": "success"
    }