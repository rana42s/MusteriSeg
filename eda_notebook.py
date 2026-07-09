#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from IPython.display import display

# 1. Veriyi yükleme
df = pd.read_csv("Customer_Data.csv")

# 2. Şema ve Boyut Doğrulama
print("--- Veri Seti Boyutu ---")
print(df.shape)

print("\n--- Veri Tipleri ve Eksik Veri Genel Bakış ---")
print(df.info())

print("\n--- İstatistiksel Özet (Metriklerin Sınırları) ---")
display(df.describe().T)


# In[2]:


df.set_index('CUST_ID', inplace=True)
print(df.head()) # Kontrol et, CUST_ID artık bir kolon değil, index.


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Eksik Veri Tespiti
missing_values = df.isnull().sum()
print("--- Kolon Bazlı Eksik Veri Sayıları ---")
print(missing_values[missing_values > 0])

# 2. MINIMUM_PAYMENTS Frekans Dağılımı (Neden eksik olduğunu anlamak için)
plt.figure(figsize=(10, 4))
sns.histplot(df['MINIMUM_PAYMENTS'].dropna(), kde=True, bins=50)
plt.title('MINIMUM_PAYMENTS Dağılım Grafiği')
plt.show()

# 3. Aykırı Değerler İçin Box-Plot Çizimi (Örn: BALANCE ve PURCHASES)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['BALANCE'])
plt.title('BALANCE Aykırı Değer Analizi')

plt.subplot(1, 2, 2)
sns.boxplot(y=df['PURCHASES'])
plt.title('PURCHASES Aykırı Değer Analizi')
plt.show()


# ### 📊 Veri Görselleştirme ve Dağılım Analiz Raporu
# 
# Bu bölümde, veri setimizdeki kritik değişkenlerin eksik veri durumları ile dağılım ve aykırı değer yapıları analiz edilmiştir.
# 
# ---
# 
# #### 1. `MINIMUM_PAYMENTS` Eksik Veri ve Dağılım Analizi
# * **Eksik Veri Sayısı:** `MINIMUM_PAYMENTS` kolonunda **313** adet eksik veri bulunmaktadır.
# * **Dağılım Yapısı (Histogram):** 
#   * Grafik incelendiğinde dağılımın **aşırı sağa çarpık (pozitif asimetrik)** olduğu görülmektedir (Çarpıklık Katsayısı: `~13.62`).
#   * Ortalama değer (`864.21`), aşırı yüksek uç değerlerin etkisiyle medyan değerinden (`312.34`) çok daha yüksektir.
#   * ⚠️ **Aksiyon Notu:** Dağılımın yüksek derecede asimetrik olması nedeniyle, 2. haftada yapacağımız eksik veri doldurma (imputation) işleminde **ortalama (mean) yerine medyan (median)** kullanmalıyız. Ortalama kullanımı verinin genel eğilimini yapay şekilde yukarı çekecektir. Bu durum ClickUp üzerinde bir aksiyon notu olarak takip edilecektir.
# 
# ---
# 
# #### 2. `BALANCE` (Bakiye) Aykırı Değer Analizi
# * **Aykırı Değer Eşiği:** $Q_3 + 1.5 \times IQR \approx 4942.93$
# * **Durum Değerlendirmesi (Box Plot):** 
#   * Müşterilerin bakiye dağılımında üst sınırın üzerinde **695** adet aykırı değer (`%7.77`) bulunmaktadır.
#   * Müşterilerin çoğunluğu daha düşük bakiyelere sahipken, az sayıda müşterinin çok yüksek bakiyeler taşıdığı görülmektedir. Bu durum, segmentasyon yaparken bakiye odaklı özel bir küme oluşabileceğine işaret eder.
# 
# ---
# 
# #### 3. `PURCHASES` (Alışveriş Tutarı) Aykırı Değer Analizi
# * **Aykırı Değer Eşiği:** $Q_3 + 1.5 \times IQR \approx 2715.88$
# * **Durum Değerlendirmesi (Box Plot):** 
#   * Alışveriş tutarlarında **808** adet aykırı değer (`%9.03`) tespit edilmiştir.
#   * Dağılım, çok az sayıda yüksek hacimli alışveriş yapan "süper kullanıcı" olduğunu göstermektedir. Bu uç değerler kümeleme algoritmalarını (örneğin K-Means) olumsuz etkileyebileceği için modelleme aşamasında log dönüşümü veya kırpma (clipping) gibi yöntemler değerlendirilmelidir.
# 

# In[4]:


# 1. Korelasyon Matrisi (Isı Haritası)
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Özellikler Arası Korelasyon Matrisi (Multicollinearity Kontrolü)')
plt.show()

# 2. Çarpıklık (Skewness) Değerlerinin Hesaplanması
print("--- Özelliklerin Çarpıklık Dereceleri ---")
print(df.skew().sort_values(ascending=False))


# ### 📊 Detaylı Korelasyon ve Dağılım (Çarpıklık) Analizi Raporu
# 
# Bu analiz adımında, veri setimizin yapısal özelliklerini anlamak, modelleme öncesinde veri ön işleme (data preprocessing) adımlarına yön vermek ve veri kalitesini artırmak amacıyla **Korelasyon Analizi** ve **Çarpıklık (Skewness) Analizi** gerçekleştirilmiştir.
# 
# ---
# 
# #### 1. Korelasyon Analizi (Neden Yapılır ve Burada Ne İşe Yarayacak?)
# 
# ##### **Korelasyon Nedir ve Neden Analiz Edilir?**
# Korelasyon, iki değişken arasındaki doğrusal ilişkinin yönünü ve gücünü ölçer. Değerler -1 ile +1 arasında değişir. Veri biliminde korelasyon analizi yapmamızın temel nedenleri şunlardır:
# * **Gereksiz Bilgileri Temizlemek:** Birbirini neredeyse mükemmel şekilde açıklayan değişkenleri tespit ederek veri setinin boyutunu azaltmak.
# * **Çoklu Doğrusallık (Multicollinearity) Sorununu Engellemek:** Yüksek korelasyonlu değişkenler modelde yer aldığında varyansı şişirir (özellikle doğrusal/regresyon modellerinde katsayıları kararsızlaştırır).
# * **Mesafe Bazlı Modelleri Korumak:** Müşteri segmentasyonunda kullanacağımız **K-Means** gibi mesafe bazlı (Euclidean distance) kümeleme algoritmalarında, yüksek derecede korelasyonlu iki değişkenin bulunması, o boyutun mesafeyi iki kat daha fazla etkilemesine (çift sayılmasına) neden olur. Bu da kümelerin hatalı veya yanlı oluşmasına yol açar.
# 
# ##### **Buradaki Bulgularımız ve Mühendislik Kararı:**
# * **`PURCHASES`** (Toplam Alışveriş Tutarı) ile **`ONEOFF_PURCHASES`** (Tek Seferlik Alışveriş Tutarı) değişkenleri arasında **0.92** (son derece yüksek) korelasyon bulunmaktadır.
# * Benzer şekilde, **`PURCHASES_FREQUENCY`** ile **`PURCHASES_INSTALLMENTS_FREQUENCY`** arasında **0.86** gibi yüksek bir korelasyon görülmektedir.
# * 💡 **Karar:** Bu değişkenler neredeyse aynı bilgiyi taşımaktadır. Modelleme aşamasında bu özellikleri doğrudan K-Means algoritmamıza sokarsak, harcama büyüklüğü ve sıklığı bilgisi mükerrer ağırlıklandırılacaktır. Bu nedenle veri setine **PCA (Principal Component Analysis - Boyut İndirgeme)** uygulamak veya bu korelasyonlu özelliklerden birini elemek **zorunludur**.
# 
# ---
# 
# #### 2. Çarpıklık (Skewness) Analizi (Neden Yapılır ve Nerede Kullanılacak?)
# 
# ##### **Çarpıklık (Skewness) Nedir?**
# Çarpıklık, bir veri dağılımının asimetri derecesini ölçer. 
# * **Pozitif (Sağa) Çarpıklık (> 0):** Dağılımın sağ kuyruğu uzundur. Verilerin büyük kısmı solda (düşük değerlerde) toplanmışken, çok az sayıda ama çok yüksek değerli uç değerler (outliers) sağ tarafta yer alır.
# * **Negatif (Sola) Çarpıklık (< 0):** Dağılımın sol kuyruğu uzundur.
# 
# ##### **Hesaplanan Değerlerimizin Dağılım Analizi:**
# Veri setimizdeki özelliklerin çarpıklık dereceleri aşağıdaki gibi hesaplanmıştır:
# * **`MINIMUM_PAYMENTS` (13.62):** Aşırı derecede sağa çarpık. Müşterilerin çoğunun asgari ödemeleri düşükken, çok az sayıda müşterinin asgari ödeme tutarı çok yüksektir.
# * **`ONEOFF_PURCHASES` (10.05) ve `PURCHASES` (8.14):** Harcama tutarlarında da çok ciddi bir pozitif çarpıklık söz konusudur.
# * **`INSTALLMENTS_PURCHASES` (7.29), `PAYMENTS` (5.91) ve `CASH_ADVANCE` (5.17):** Finansal harcama ve nakit avans hareketleri de benzer şekilde sağa çarpıktır.
# * **`BALANCE_FREQUENCY` (-2.02) ve `TENURE` (-2.94):** Negatif çarpıklığa sahiptir. Müşterilerin çoğunun bakiyelerini çok sık güncellediğini ve genellikle 12 aylık (maksimum) vadeye sahip olduğunu gösterir.
# 
# ##### **Nerede ve Nasıl Kullanacağız? (Aksiyon Planı)**
# 
# 1. **Eksik Veri Doldurma (Imputation) Stratejisi:**
#    * `MINIMUM_PAYMENTS` kolonunda **313** adet eksik veri bulunmaktadır.
#    * Bu değişkenin çarpıklık katsayısı **13.62** (aşırı çarpık) olduğu için eksik değerler kesinlikle **ortalama (mean)** ile doldurulmamalıdır. Çünkü ortalama, uç değerler tarafından yukarı çekilmiştir ve veriyi yapay olarak bozacaktır. Doldurma işlemi için **medyan (median)** değeri kullanılacaktır.
#    
# 2. **Logaritmik Dönüşüm (Transformation) İhtiyacı:**
#    * K-Means ve PCA gibi algoritmalar, değişkenlerin nispeten simetrik (normal dağılıma yakın) olmasını ve uç değerlerin mesafeyi domine etmemesini bekler.
#    * Çarpıklık katsayısı **1 veya 2'nin üzerinde olan finansal kolonlara** (örn. `BALANCE`, `PURCHASES`, `PAYMENTS`, `MINIMUM_PAYMENTS`, `CASH_ADVANCE`) modelleme öncesinde **Log Dönüşümü ($log(x+1)$)** uygulanacaktır. Bu işlem çarpıklığı azaltarak dağılımları normal dağılıma yaklaştıracak ve kümeleme başarısını doğrudan artıracaktır.

# ### 🛠️ Veri Ön İşleme (Data Preprocessing): Eksik Veri Doldurma ve Logaritmik Dönüşüm
# 
# Bu adımda, önceki hücrelerde yaptığımız dağılım ve çarpıklık (skewness) analizleri doğrultusunda verilerimizi modellemeye hazırlıyoruz:
# 
# ---
# 
# #### 1. Eksik Veri İmputasyonu (Data Imputation)
# * **`MINIMUM_PAYMENTS` (313 eksik değer):** Bu kolonun dağılımı aşırı derecede sağa çarpık (`~13.85`) olduğundan, eksik verileri ortalama (mean) yerine **medyan (median)** değeri ile dolduruyoruz. Ortalama kullanımı, uç değerlerin etkisiyle veriyi yapay olarak yukarı çekebilirdi.
# * **`CREDIT_LIMIT` (1 eksik değer):** Benzer şekilde bu kolondaki tek eksik değeri de **medyan** değeriyle doldurarak veri bütünlüğünü sağlıyoruz.
# 
# ---
# 
# #### 2. Matematiksel Dönüşümler (Log Transformation)
# * **Neden Log Dönüşümü Yapıyoruz?**
#   * K-Means ve PCA gibi algoritmalar, özelliklerin dağılımının dengeli ve simetrik olmasını bekler. Aşırı sağa çarpık finansal değişkenler, mesafe bazlı uzaklık hesaplamalarında baskınlık kurarak kümeleme başarısını olumsuz etkiler.
# * **Hangi Kolonlara Uyguluyoruz?**
#   * Çarpıklık derecesi yüksek olan tüm harcama ve bakiye kolonlarına: `BALANCE`, `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`, `PAYMENTS`, `MINIMUM_PAYMENTS`.
# * **Kullanılan Yöntem:**
#   * **$log(x+1)$** (kodda `np.log1p`): Müşterilerimizin bazı kolonlarda (örneğin nakit avans veya tek seferlik alışveriş tutarı) sıfır (`0`) harcamaları bulunmaktadır. Doğrudan logaritma almak tanımsızlık yaratacağından, $log(x+1)$ kullanarak sıfır değerleri koruyor ve tüm pozitif değerleri başarıyla normalize ediyoruz.
# 
# Aşağıdaki kod hücresinde bu işlemlerin öncesi ve sonrası çarpıklık dereceleri ile histogram grafikleri karşılaştırmalı olarak çizilmiştir.

# In[5]:


# 1. Eksik Değerleri Medyan ile Doldurma
df_clean = df.copy()
df_clean['MINIMUM_PAYMENTS'] = df_clean['MINIMUM_PAYMENTS'].fillna(df_clean['MINIMUM_PAYMENTS'].median())
df_clean['CREDIT_LIMIT'] = df_clean['CREDIT_LIMIT'].fillna(df_clean['CREDIT_LIMIT'].median())

print("Eksik veri doldurma sonrası kontrol:")
print(df_clean[['MINIMUM_PAYMENTS', 'CREDIT_LIMIT']].isnull().sum())

# 2. Logaritmik Dönüşüm (np.log1p)
cols_to_log = ['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS']

print("\nDönüşüm Öncesi Çarpıklık (Skewness):")
print(df_clean[cols_to_log].skew().sort_values(ascending=False))

df_transformed = df_clean.copy()
for col in cols_to_log:
    df_transformed[col] = np.log1p(df_transformed[col])

print("\nDönüşüm Sonrası Çarpıklık (Skewness):")
print(df_transformed[cols_to_log].skew().sort_values(ascending=False))

# 3. Görsel Karşılaştırma (Orijinal vs Log Dönüştürülmüş)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 10))

# BALANCE Karşılaştırma
plt.subplot(2, 2, 1)
sns.histplot(df_clean['BALANCE'], kde=True, bins=30, color='blue')
plt.title('BALANCE - Orijinal Dağılım')

plt.subplot(2, 2, 2)
sns.histplot(df_transformed['BALANCE'], kde=True, bins=30, color='green')
plt.title('BALANCE - Log Dönüşümlü Dağılım')

# PURCHASES Karşılaştırma
plt.subplot(2, 2, 3)
sns.histplot(df_clean['PURCHASES'], kde=True, bins=30, color='blue')
plt.title('PURCHASES - Orijinal Dağılım')

plt.subplot(2, 2, 4)
sns.histplot(df_transformed['PURCHASES'], kde=True, bins=30, color='green')
plt.title('PURCHASES - Log Dönüşümlü Dağılım')

plt.tight_layout()
plt.show()


