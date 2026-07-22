import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# 1. Sayfa Yapılandırması ve Kurumsal Tema
st.set_page_config(
    page_title="BKM Segmentasyon Analizörü",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Kredi Kartı Müşteri Segmentasyon Paneli")
st.markdown("Bu panel, müşteri finansal hareketlerini analiz ederek yapay zeka destekli segmentasyon ve ürün önerisi sunar.")
st.markdown("---")

# 2. API Adresleri (FastAPI sunucumuzun adresleri)
API_URL_SINGLE = "http://127.0.0.1:8000/predict"
API_URL_BATCH = "http://127.0.0.1:8000/predict_batch"

# SAYFAYI İKİ SEKMEYE (TAB) BÖLÜYORUZ
tab1, tab2 = st.tabs(["👤 Tekil Müşteri Analizi", "📁 Toplu Müşteri Analizi (CSV Yükleme)"])

# ==========================================
# SEKME 1: TEKİL MÜŞTERİ ANALİZİ
# ==========================================
with tab1:
    st.subheader("👤 Bireysel Müşteri Finansal Verileri")
    
    # Form key değerini benzersiz yaptık
    with st.form(key="single_customer_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            balance = st.number_input("Bakiye (BALANCE)", min_value=0.0, value=8500.0, step=100.0)
            balance_freq = st.number_input("Bakiye Güncelleme Sıklığı", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
            purchases = st.number_input("Toplam Harcama (PURCHASES)", min_value=0.0, value=45000.0, step=100.0)
            oneoff = st.number_input("Tek Çekim Harcama (ONEOFF)", min_value=0.0, value=45000.0, step=100.0)
            installments = st.number_input("Taksitli Harcama (INSTALLMENTS)", min_value=0.0, value=0.0, step=100.0)
            cash_advance = st.number_input("Nakit Avans (CASH_ADVANCE)", min_value=0.0, value=0.0, step=100.0)
            
        with col2:
            purchases_freq = st.number_input("Harcama Sıklığı", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
            oneoff_freq = st.number_input("Tek Çekim Sıklığı", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
            installments_freq = st.number_input("Taksit Sıklığı", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
            cash_freq = st.number_input("Nakit Avans Sıklığı", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
            cash_trx = st.number_input("Nakit Avans İşlem Adedi", min_value=0, value=0, step=1)
            purchases_trx = st.number_input("Alışveriş İşlem Adedi", min_value=0, value=85, step=1)
            
        with col3:
            credit_limit = st.number_input("Kredi Limiti (CREDIT_LIMIT)", min_value=0.0, value=100000.0, step=1000.0)
            payments = st.number_input("Yapılan Ödemeler (PAYMENTS)", min_value=0.0, value=42000.0, step=100.0)
            min_payments = st.number_input("Asgari Ödeme (MIN_PAYMENTS)", min_value=0.0, value=1500.0, step=100.0)
            prc_full_payment = st.number_input("Borcun Tamamını Ödeme Oranı", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
            tenure = st.number_input("Müşteri Yaşı (Ay)", min_value=6, max_value=12, value=12, step=1)

        submit_button = st.form_submit_button(label="🚀 Müşteriyi Analiz Et")

    if submit_button:
        payload = {
            "BALANCE": balance, "BALANCE_FREQUENCY": balance_freq, "PURCHASES": purchases,
            "ONEOFF_PURCHASES": oneoff, "INSTALLMENTS_PURCHASES": installments, 
            "CASH_ADVANCE": cash_advance, "PURCHASES_FREQUENCY": purchases_freq,
            "ONEOFF_PURCHASES_FREQUENCY": oneoff_freq, "PURCHASES_INSTALLMENTS_FREQUENCY": installments_freq,
            "CASH_ADVANCE_FREQUENCY": cash_freq, "CASH_ADVANCE_TRX": cash_trx,
            "PURCHASES_TRX": purchases_trx, "CREDIT_LIMIT": credit_limit,
            "PAYMENTS": payments, "MINIMUM_PAYMENTS": min_payments,
            "PRC_FULL_PAYMENT": prc_full_payment, "TENURE": tenure
        }
        with st.spinner('Yapay Zeka Modeli Müşteriyi Analiz Ediyor...'):
            try:
                response = requests.post(API_URL_SINGLE, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    persona = result["persona"]
                    cluster_id = result["cluster_id"]
                    
                    st.success("✅ Analiz Başarıyla Tamamlandı!")
                    
                    st.markdown("### 🎯 Segmentasyon Sonucu")
                    st.info(f"**Atanan Küme:** Cluster {cluster_id} \n\n **Müşteri Profili:** {persona}")
                    
                    st.markdown("### 💡 Aksiyon / Ürün Önerisi")
                    if cluster_id == 0:
                        st.success("VIP Müşteri Temsilcisi atayın. Mil kazandıran kredi kartı (Platinum) teklif edin.")
                    elif cluster_id == 1:
                        st.warning("Churn riski! İlk 1000 TL harcamaya 200 TL chip-para kampanyası SMS'i gönderin.")
                    elif cluster_id == 2:
                        st.success("Taksit sever profil. Elektronik alışverişlerinde +3 Taksit kampanyası tanımlayın.")
                    elif cluster_id == 3:
                        st.error("Yüksek nakit borçlanma tespiti! Kredi kartı limit artışını durdurun, Taksitli Nakit Avans yapılandırması önerin.")
                else:
                    st.error(f"API Hatası: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("⚠️ Sunucuya bağlanılamadı. Lütfen FastAPI sunucusunun (api.py) arka planda çalıştığından emin olun.")

# ==========================================
# SEKME 2: TOPLU MÜŞTERİ ANALİZİ VE 3B GÖRSELLEŞTİRME
# ==========================================
with tab2:
    st.subheader("📁 Toplu Müşteri Verisi Yükleme (CSV)")
    st.info("Elinizdeki ham müşteri veri setini yükleyerek anında segmentasyon yapın ve veri uzayını görselleştirin.")
    
    uploaded_file = st.file_uploader("Müşteri Verisi (CSV) Seçin", type=["csv"])
    
    if uploaded_file is not None:
        if st.button("Toplu Analizi Başlat 🚀", key="batch_btn"):
            with st.spinner("Büyük veri işleniyor ve kümeler atanıyor. Lütfen bekleyin..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    response_batch = requests.post(API_URL_BATCH, files=files)
                    
                    if response_batch.status_code == 200:
                        result_data = response_batch.json()["data"]
                        df_result = pd.DataFrame(result_data)
                        
                        st.success(f"✅ Başarılı! {len(df_result)} adet müşterinin segmentasyon işlemi tamamlandı.")
                        
                        st.markdown("### 📊 Segmentasyon Sonuç Tablosu")
                        csv_export = df_result.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Etiketlenmiş Veriyi İndir (CSV)",
                            data=csv_export,
                            file_name='segmentasyon_tamamlandi.csv',
                            mime='text/csv'
                        )
                        st.dataframe(df_result.head(100))
                        
                        st.markdown("### 🌌 3 Boyutlu Müşteri Kümeleme Uzayı")
                        st.markdown("Grafiği farenizle döndürebilir, yakınlaştırabilir ve noktaların üzerine gelerek detayları görebilirsiniz.")
                        
                        fig = px.scatter_3d(
                            df_result,
                            x='BALANCE', 
                            y='PURCHASES', 
                            z='CASH_ADVANCE', 
                            color='PERSONA', 
                            hover_data=['CREDIT_LIMIT'], 
                            opacity=0.7,
                            title="Müşteri Finansal Uzayında K-Means Dağılımı",
                            color_discrete_sequence=px.colors.qualitative.Bold
                        )
                        fig.update_layout(height=700) 
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.error(f"API İşlem Hatası: {response_batch.json()['detail']}")
                except Exception as e:
                    st.error(f"Bağlantı veya Kod Hatası: {str(e)}")