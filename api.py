from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import io
import json

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

# 5. Batch Tahmin Yapan Endpoint'i Kodluyoruz - Tekil Tahminin Aksine, CSV Dosyası ile Çoklu Tahmin
@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    # 1. Dosya uzantısı kontrolü
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Hatalı format. Lütfen sadece CSV dosyası yükleyin.")
    
    try:
        # 2. Dosyayı hafızaya okuma (Fiziksel kayıt yapmadan)
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 3. Modele girecek veriyi hazırlama
        # (Eğer veride CUST_ID varsa model hata verebilir, onu geçici olarak ayırıyoruz)
        if "CUST_ID" in df.columns:
            df_features = df.drop("CUST_ID", axis=1)
        else:
            df_features = df.copy()
            
        # 4. Pipeline ve K-Means İşlemleri
        X_scaled = preprocessing_pipeline.transform(df_features)
        clusters = kmeans_model.predict(X_scaled)
        
        # 5. Sonuçları orijinal veri setine kolon olarak ekleme
        df["CLUSTER_ID"] = clusters
        
        # 6. Müşteri profillerini (Persona) atama
        persona_map = {
            0: "Premium / Elit Harcamacılar",
            1: "İnaktif / Pasif Müşteriler",
            2: "Bütçe Odaklı Taksit Severler",
            3: "Nakit Avans Bağımlıları"
        }
        df["PERSONA"] = df["CLUSTER_ID"].map(persona_map)
        
        # 7. Streamlit'in doğrudan okuyabileceği "records" formatında JSON döndürme
        # df.to_dict() yerine pandas'ın daha güvenli olan kendi JSON dönüştürücüsünü kullanıyoruz:
        parsed_data = json.loads(df.to_json(orient="records"))
        return {"data": parsed_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Veri işlenirken bir hata oluştu: {str(e)}")