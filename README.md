# Kredi Kartı Kullanıcı Segmentasyonu (Credit Card Customer Segmentation)

Bu proje, bir bankanın aktif kredi kartı müşterilerinin kullanım alışkanlıklarını analiz ederek, pazarlama stratejilerini optimize etmek amacıyla müşterileri benzer davranış gruplarına (segmentlere) ayırmayı hedeflemektedir.

Proje kapsamında gerçekleştirilen veri analizi (EDA) ve ön işleme adımları [eda_notebook.ipynb](eda_notebook.ipynb) içerisinde detaylandırılmıştır.

---

## 📊 Veri Seti Hakkında

Projede kullanılan `Customer_Data.csv` veri seti, son 6 aydır aktif olan **8950** kredi kartı müşterisinin kart kullanım davranışlarını ve 18 farklı özniteliğini (feature) içermektedir.

### Öznitelik Tanımları

*   `CUST_ID`: Müşteri Benzersiz Kimliği
*   `BALANCE`: Hesapta kalan bakiye tutarı
*   `BALANCE_FREQUENCY`: Bakiyenin güncellenme sıklığı (0-1 arası)
*   `PURCHASES`: Yapılan toplam alışveriş tutarı
*   `ONEOFF_PURCHASES`: Tek seferde yapılan en yüksek alışveriş tutarı
*   `INSTALLMENTS_PURCHASES`: Taksitli yapılan alışveriş tutarı
*   `CASH_ADVANCE`: Nakit avans çekim tutarı
*   `PURCHASES_FREQUENCY`: Alışveriş yapma sıklığı (0-1 arası)
*   `ONEOFF_PURCHASES_FREQUENCY`: Tek seferlik alışveriş sıklığı (0-1 arası)
*   `PURCHASES_INSTALLMENTS_FREQUENCY`: Taksitli alışveriş sıklığı (0-1 arası)
*   `CASH_ADVANCE_FREQUENCY`: Nakit avans çekim sıklığı (0-1 arası)
*   `CASH_ADVANCE_TRX`: Nakit avans çekim işlem sayısı
*   `PURCHASES_TRX`: Alışveriş işlem sayısı
*   `CREDIT_LIMIT`: Kredi kartı limiti
*   `PAYMENTS`: Yapılan toplam ödeme tutarı
*   `MINIMUM_PAYMENTS`: Yapılan asgari ödeme tutarı
*   `PRC_FULL_PAYMENT`: Borcun tamamını ödeyenlerin oranı
*   `TENURE`: Kredi kartı sahiplik süresi (ay)

---

## 🔍 Keşifçi Veri Analizi (EDA) Bulguları

Notebook üzerinde yapılan analizlerde şu önemli yapısal bulgular elde edilmiştir:

1.  **Aykırı Değerler (Outliers):**
    *   `BALANCE` kolonu için müşterilerin %7.77'si (695 müşteri) bakiye sınırının üzerinde aykırı değer taşımaktadır.
    *   `PURCHASES` (Alışveriş Tutarı) kolonu için %9.03'lük (808 müşteri) bir süper-kullanıcı (uç değerler) kitlesi tespit edilmiştir.
    *   Bu uç değerlerin kümeleme modellerinin (örneğin K-Means) mesafe hesaplamalarını bozmaması için log dönüşümü veya kırpma (clipping) uygulanması önerilmektedir.

2.  **Korelasyon Analizi (Multicollinearity):**
    *   `PURCHASES` ile `ONEOFF_PURCHASES` arasında **0.92** oranında son derece yüksek bir doğrusal ilişki saptanmıştır.
    *   `PURCHASES_FREQUENCY` ile `PURCHASES_INSTALLMENTS_FREQUENCY` arasında **0.86** korelasyon bulunmaktadır.
    *   *Karar:* Kümeleme modellerinde mükerrer ağırlık oluşturmaması amacıyla modelleme öncesinde **PCA (Temel Bileşenler Analizi)** ile boyut indirgeme yapılması veya korelasyonlu özelliklerden birinin elenmesi gereklidir.

3.  **Çarpıklık (Skewness) Değerlendirmesi:**
    *   `MINIMUM_PAYMENTS` (13.62), `ONEOFF_PURCHASES` (10.05) ve `PURCHASES` (8.14) öznitelikleri aşırı derecede sağa çarpık (pozitif skewness) bir dağılıma sahiptir.
    *   Buna karşılık, müşterilerin çoğu bakiyelerini çok sık güncellediği için `BALANCE_FREQUENCY` (-2.02) negatif çarpıklığa sahiptir.

---

## 🛠️ Veri Ön İşleme (Preprocessing) ve Dönüşümler

Analiz sonuçlarına dayanarak veri kalitesini ve model başarısını artırmak için şu ön işleme adımları uygulanmıştır:

*   **Eksik Veri Doldurma (Imputation):** 
    *   `MINIMUM_PAYMENTS` kolonundaki 313 eksik değer, verinin yüksek çarpıklığı nedeniyle ortalama yerine **medyan (median)** ile doldurulmuştur.
    *   `CREDIT_LIMIT` kolonundaki 1 eksik değer yine medyan ile tamamlanmıştır.
*   **Matematiksel Dönüşümler (Log Transformation):**
    *   Çarpıklığı gidermek ve özellikleri normal dağılıma yaklaştırmak için finansal değişkenlere (bakiye, harcama vb.) **$log(x+1)$** (`np.log1p`) dönüşümü uygulanmıştır. Sıfır değerli harcamalar bu dönüşüm sayesinde korunmuştur.

---

## 🚀 Kurulum ve Çalıştırma

Projenin bağımlılıklarını yerel ortamınıza yüklemek için:

```bash
# Depoyu klonlayın
git clone <github-repo-url>
cd MusteriSeg

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt
```

Ardından Jupyter Lab/Notebook veya VS Code üzerinden [eda_notebook.ipynb](eda_notebook.ipynb) dosyasını açıp tüm hücreleri çalıştırabilirsiniz.
