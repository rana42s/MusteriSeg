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
    
    # ==========================================
    # 1. SIFIRLAMA VE ÖRNEK PROFİL YÖNETİMİ
    # ==========================================
    default_values = {
        "balance": 8500.0, "balance_freq": 1.0, "purchases": 45000.0,
        "oneoff": 45000.0, "installments": 0.0, "cash_advance": 0.0,
        "purchases_freq": 1.0, "oneoff_freq": 1.0, "installments_freq": 0.0,
        "cash_freq": 0.0, "cash_trx": 0, "purchases_trx": 85,
        "credit_limit": 100000.0, "payments": 42000.0, "min_payments": 1500.0,
        "prc_full_payment": 0.9, "tenure": 12
    }

    # Session state içinde değerler yoksa başlatıyoruz
    for key, val in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = val
            
    if "profile_desc" not in st.session_state:
        st.session_state["profile_desc"] = None

    st.markdown("### 🔍 Örnek Müşteri Profillerini İnceleyin")
    st.caption("Aşağıdaki butonlara tıklayarak yapay zekanın farklı harcama alışkanlıklarını nasıl kümelediğini test edebilirsiniz.")

    # 4 Profil butonu ve 1 Sıfırlama butonu için 5 sütun oluşturuyoruz
    col_p1, col_p2, col_p3, col_p4, col_reset = st.columns(5)
    
    with col_p1:
        if st.button("👑 Premium Müşteri"):
            st.session_state.update({
                "balance": 25000.0, "balance_freq": 1.0, "purchases": 120000.0,
                "oneoff": 100000.0, "installments": 20000.0, "cash_advance": 0.0,
                "purchases_freq": 1.0, "oneoff_freq": 0.9, "installments_freq": 0.5,
                "cash_freq": 0.0, "cash_trx": 0, "purchases_trx": 150,
                "credit_limit": 300000.0, "payments": 115000.0, "min_payments": 3000.0,
                "prc_full_payment": 1.0, "tenure": 12,
                "profile_desc": "**👑 Premium / Elit Harcamacılar:** Kredi kartı limiti çok yüksektir. Harcamalarının büyük çoğunluğunu tek çekim (ONEOFF) yapar ve ekstre borcunun tamamını (PRC_FULL_PAYMENT = 1.0) düzenli öderler. Nakit avans kullanmazlar."
            })
            st.rerun()

    with col_p2:
        if st.button("💤 İnaktif Müşteri"):
            st.session_state.update({
                "balance": 150.0, "balance_freq": 0.2, "purchases": 0.0,
                "oneoff": 0.0, "installments": 0.0, "cash_advance": 0.0,
                "purchases_freq": 0.0, "oneoff_freq": 0.0, "installments_freq": 0.0,
                "cash_freq": 0.0, "cash_trx": 0, "purchases_trx": 0,
                "credit_limit": 10000.0, "payments": 0.0, "min_payments": 0.0,
                "prc_full_payment": 0.0, "tenure": 12,
                "profile_desc": "**💤 İnaktif / Cüzdanda Unutanlar:** Kartı uzun süredir ellerinde bulunmasına rağmen (TENURE=12) aktif olarak kullanmazlar. Bakiye, harcama ve işlem sıklıkları sıfıra yakındır. Churn (terk) riski en yüksek gruptur."
            })
            st.rerun()

    with col_p3:
        if st.button("🛍️ Taksit Sever"):
            st.session_state.update({
                "balance": 1500.0, "balance_freq": 1.0, "purchases": 2500.0,
                "oneoff": 300.0, "installments": 2200.0, "cash_advance": 0.0,
                "purchases_freq": 0.9, "oneoff_freq": 0.1, "installments_freq": 0.9,
                "cash_freq": 0.0, "cash_trx": 0, "purchases_trx": 15,
                "credit_limit": 6000.0, "payments": 1200.0, "min_payments": 500.0,
                "prc_full_payment": 0.2, "tenure": 12,
                "profile_desc": "**🛍️ Bütçe Odaklı Taksit Severler:** Toplam harcamalarının büyük kısmı taksitli alışverişlerden (INSTALLMENTS) oluşur. Taksit yapma sıklıkları yüksektir. Harcamalarını aylara bölerek bütçelerini kontrol etmeyi severler."
            })
            st.rerun()

    with col_p4:
        if st.button("💸 Nakit Avansçı"):
            st.session_state.update({
                "balance": 4500.0, "balance_freq": 1.0, "purchases": 150.0,
                "oneoff": 150.0, "installments": 0.0, "cash_advance": 3800.0,
                "purchases_freq": 0.1, "oneoff_freq": 0.1, "installments_freq": 0.0,
                "cash_freq": 0.9, "cash_trx": 8, "purchases_trx": 2,
                "credit_limit": 5000.0, "payments": 600.0, "min_payments": 1500.0,
                "prc_full_payment": 0.0, "tenure": 12,
                "profile_desc": "**💸 Nakit Avans Bağımlıları:** Nakit avans (CASH_ADVANCE) çekim tutarları ve sıklıkları çok yüksektir. Ekstre borçlarının tamamını ödeme oranları çok düşüktür (genelde sadece asgari tutarı öderler). Yüksek faiz yükü altındadırlar."
            })
            st.rerun()
            
    with col_reset:
        if st.button("🔄 Sıfırla"):
            for key, val in default_values.items():
                st.session_state[key] = val
            st.session_state["profile_desc"] = None
            st.rerun()

    # Tıklanan profilin açıklamasını formun üstünde gösteriyoruz
    if st.session_state["profile_desc"]:
        st.info(st.session_state["profile_desc"])
    
    st.divider()
    
    # ==========================================
    # 2. KULLANICI GİRİŞ FORMU
    # ==========================================
    with st.form(key="single_customer_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            balance = st.number_input("Bakiye (BALANCE)", min_value=0.0, value=st.session_state["balance"], step=100.0, key="balance", help="Müşterinin hesabında bulunan mevcut bakiye.")
            balance_freq = st.number_input("Bakiye Güncelleme Sıklığı", value=st.session_state["balance_freq"], step=0.1, key="balance_freq", help="0 ile 1 arasında bir değer (Örn: 1.0 = Sık güncelleniyor)")
            purchases = st.number_input("Toplam Harcama (PURCHASES)", min_value=0.0, value=st.session_state["purchases"], step=100.0, key="purchases")
            oneoff = st.number_input("Tek Çekim Harcama (ONEOFF)", min_value=0.0, value=st.session_state["oneoff"], step=100.0, key="oneoff")
            installments = st.number_input("Taksitli Harcama (INSTALLMENTS)", min_value=0.0, value=st.session_state["installments"], step=100.0, key="installments")
            cash_advance = st.number_input("Nakit Avans (CASH_ADVANCE)", min_value=0.0, value=st.session_state["cash_advance"], step=100.0, key="cash_advance")
            
        with col2:
            purchases_freq = st.number_input("Harcama Sıklığı", value=st.session_state["purchases_freq"], step=0.1, key="purchases_freq")
            oneoff_freq = st.number_input("Tek Çekim Sıklığı", value=st.session_state["oneoff_freq"], step=0.1, key="oneoff_freq")
            installments_freq = st.number_input("Taksit Sıklığı", value=st.session_state["installments_freq"], step=0.1, key="installments_freq")
            cash_freq = st.number_input("Nakit Avans Sıklığı", value=st.session_state["cash_freq"], step=0.1, key="cash_freq")
            cash_trx = st.number_input("Nakit Avans İşlem Adedi", min_value=0, value=st.session_state["cash_trx"], step=1, key="cash_trx")
            purchases_trx = st.number_input("Alışveriş İşlem Adedi", min_value=0, value=st.session_state["purchases_trx"], step=1, key="purchases_trx")
            
        with col3:
            credit_limit = st.number_input("Kredi Limiti (CREDIT_LIMIT)", min_value=0.0, value=st.session_state["credit_limit"], step=1000.0, key="credit_limit")
            payments = st.number_input("Yapılan Ödemeler (PAYMENTS)", min_value=0.0, value=st.session_state["payments"], step=100.0, key="payments")
            min_payments = st.number_input("Asgari Ödeme (MIN_PAYMENTS)", min_value=0.0, value=st.session_state["min_payments"], step=100.0, key="min_payments")
            prc_full_payment = st.number_input("Borcun Tamamını Ödeme Oranı", value=st.session_state["prc_full_payment"], step=0.1, key="prc_full_payment")
            tenure = st.number_input("Müşteri Yaşı (Ay)", min_value=6, max_value=12, value=st.session_state["tenure"], step=1, key="tenure")

        submit_button = st.form_submit_button(label="🚀 Müşteriyi Analiz Et")

    # ==========================================
    # 3. MANUEL VERİ DOĞRULAMA VE API İSTEĞİ
    # ==========================================
    if submit_button:
        freq_fields = {
            "Bakiye Güncelleme Sıklığı": balance_freq,
            "Harcama Sıklığı": purchases_freq,
            "Tek Çekim Sıklığı": oneoff_freq,
            "Taksit Sıklığı": installments_freq,
            "Nakit Avans Sıklığı": cash_freq,
            "Borcun Tamamını Ödeme Oranı": prc_full_payment
        }
        
        invalid_fields = [name for name, val in freq_fields.items() if not (0.0 <= val <= 1.0)]
        is_math_invalid = round(purchases, 2) != round(oneoff + installments, 2)

        if invalid_fields:
            st.error(f"⛔ **Hata:** Aşağıdaki alanlar **0 ile 1** arasında olmalıdır:\n\n" + ", ".join(invalid_fields))
        elif is_math_invalid:
            st.error(f"⛔ **Mantıksal Hata:** Toplam Harcama ({purchases}), Tek Çekim ({oneoff}) ve Taksitli Harcamaların ({installments}) toplamına eşit olmalıdır.")
        else:
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
                        try:
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
                        except ValueError:
                            st.error("⚠️ API'den geçersiz bir format döndü. (JSON çözümlenemedi)")
                    else:
                        st.error(f"API Hatası: {response.status_code} - Sunucu isteği işleyemedi.")
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