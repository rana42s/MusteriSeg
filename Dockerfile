FROM python:3.12-slim

# Çalışma dizininir
WORKDIR /app

# Gereksinimler
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Sürüm çakışmasını önlemek için modeli eğittiğin sürümü sabitliyoruz
RUN pip install scikit-learn==1.6.1

# Tüm proje dosyalarını kopyalama
COPY . .

# Hugging Face Spaces dışarıya varsayılan olarak 7860 portunu açar
EXPOSE 7860

# FastAPI'yi arka planda (8000), Streamlit'i ön planda (7860) çalıştır
CMD uvicorn api:app --host 127.0.0.1 --port 8000 & streamlit run app.py --server.port 7860 --server.address 0.0.0.0
