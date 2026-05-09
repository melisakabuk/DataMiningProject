import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re

# 1. Veri Setini Yükleme
print("Veri seti yükleniyor...")
dataset = load_dataset("savasy/ttc4900")
df = pd.DataFrame(dataset['train'])

# 2. Temizlik Fonksiyonu
def temizle(metin):
    metin = metin.lower()
    metin = re.sub(r'[^\w\s]', '', metin)
    metin = re.sub(r'\d+', '', metin)
    return metin

print("Metinler temizleniyor...")
df['text_clean'] = df['text'].apply(temizle)

# 3. Veriyi Eğitim ve Test Olarak Ayırma
X_train, X_test, y_train, y_test = train_test_split(
    df['text_clean'], 
    df['category'], 
    test_size=0.2, 
    random_state=42
)

# 4. Vektörizasyon (TF-IDF)
print("Vektörizasyon yapılıyor...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Modelleri Tanımlama ve Eğitme
modeller = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

print("\n--- Model Eğitim ve Test Sonuçları ---")

for model_adi, model in modeller.items():
    # Modeli eğit
    model.fit(X_train_tfidf, y_train)
    # Tahmin yap
    tahminler = model.predict(X_test_tfidf)
    # Başarıyı ölç
    skor = accuracy_score(y_test, tahminler)
    print(f"{model_adi} Başarı Skoru: %{skor*100:.2f}")

print("\nİşlem tamamlandı!")

# --- CANLI TEST BÖLÜMÜ ---
print("\n" + "="*30)
print("CANLI TEST SİSTEMİ")
print("="*30)

def kategori_ismini_getir(no):
    sozluk = {
        0: "Siyaset", 1: "Dünya", 2: "Ekonomi", 
        3: "Kültür-Sanat", 4: "Sağlık", 5: "Spor", 6: "Teknoloji"
    }
    return sozluk.get(no, "Bilinmiyor")

while True:
    kullanici_metni = input("\nTest için bir haber cümlesi girin (Çıkmak için 'q' basın): ")
    if kullanici_metni.lower() == 'q':
        break
    
    temiz_metin = temizle(kullanici_metni)
    vektor = vectorizer.transform([temiz_metin])
    # En yüksek puanı alan modelimizi (Logistic Regression) kullanıyoruz
    tahmin_no = modeller["Logistic Regression"].predict(vektor)[0]
    
    print(f"Tahmin Edilen Kategori: {kategori_ismini_getir(tahmin_no)}")