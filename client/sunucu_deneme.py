import requests
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import json

# Veri setini yükle
def load_data(client_data_path):
    data = pd.read_csv(client_data_path)
    X = data.drop(columns=["Diabetes_binary"])  # Özellikler
    y = data["Diabetes_binary"]  # Hedef değişken

    # Veriyi eğitim ve test olarak bölelim (%80 eğitim, %20 test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Verileri ölçeklendirme
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Model oluşturma
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Yerel eğitimi yap
def local_training(X_train, y_train, X_test, y_test):
    model = create_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)
    model_weights = model.get_weights()  # Modelin ağırlıklarını al

    # Modelin performansını değerlendirme (test seti üzerinde)
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)

    # Confusion Matrix ve Classification Report yazdırma
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Performans sonuçlarını ekrana yazdır
    print("\nTest Seti Üzerindeki Doğruluk: {:.2f}%".format(accuracy * 100))
    print("\nConfusion Matrix (Karışıklık Matrisi):")
    print(cm)
    print("\nClassification Report (Sınıflandırma Raporu):")
    print(report)

    return model_weights, accuracy, model

# Global ağırlıkları sunucuya gönder
def send_weights_to_api(weights, model_type, aggregation_method):
    url = "http://34.118.207.217:5000/send_weights"
    
    # NumPy dizisini listeye dönüştürme
    weights_list = [w.tolist() for w in weights]

    data = {
        "weights": weights_list,
        "model_type": model_type,
        "aggregation_method": aggregation_method
    }
    
    response = requests.post(url, json=data)
    
    try:
        global_weights = response.json().get("global_weights")
    except ValueError:
        print("Sunucudan geçerli bir JSON yanıtı alınamadı.")
        global_weights = None
    
    return global_weights

# Global ağırlıkları yerel model ile değiştir
def update_local_model_with_global_weights(model, global_weights):
    if global_weights:
        model.set_weights([np.array(w) for w in global_weights])
        print("Model güncellendi.")

# Global ağırlıklarla modeli test et ve performansı yazdır
def evaluate_updated_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Performans sonuçlarını ekrana yazdır
    print("\nGüncellenmiş Modelin Test Seti Üzerindeki Doğruluğu: {:.2f}%".format(accuracy * 100))
    print("\nConfusion Matrix (Karışıklık Matrisi):")
    print(cm)
    print("\nClassification Report (Sınıflandırma Raporu):")
    print(report)

# Ana menü
def display_menu():
    print("\n===== Ana Menü =====")
    print("1. Modeli eğit")
    print("2. Modeli Ana sunucuya gönder")
    print("3. Ana sunucudan modeli al ve güncelle")
    print("4. Güncellenmiş modelin performansını değerlendir")
    print("5. Çıkış")

# Main fonksiyonu
if __name__ == "__main__":
    # Müşteri için veri setini yükleyelim
    client_data_path = "data.csv"  # Client 1 için veri seti yolu
    X_train, X_test, y_train, y_test = load_data(client_data_path)
    
    model_weights = None
    global_weights = None
    accuracy = None
    local_model = None

    while True:
        display_menu()
        choice = input("\nBir seçenek girin: ")

        if choice == "1":
            # Modeli eğit
            model_weights, accuracy, local_model = local_training(X_train, y_train, X_test, y_test)
            print("Model başarıyla eğitildi.")
        elif choice == "2":
            # Modeli Ana sunucuya gönder
            if model_weights:
                model_type = "dense"
                aggregation_method = "fedavg"
                send_weights_to_api(model_weights, model_type, aggregation_method)
                print("Model ağırlıkları Ana sunucuya gönderildi.")
            else:
                print("Model henüz eğitilmedi. Lütfen önce modeli eğitin.")
        elif choice == "3":
            # Ana sunucudan modeli al ve yerel modeli güncelle
            if model_weights:
                global_weights = send_weights_to_api(model_weights, "dense", "fedavg")
                if global_weights:
                    update_local_model_with_global_weights(local_model, global_weights)
                else:
                    print("Sunucudan model ağırlıkları alınamadı.")
            else:
                print("Model henüz eğitilmedi. Lütfen önce modeli eğitin.")
        elif choice == "4":
            # Güncellenmiş modelin performansını değerlendir
            if local_model and global_weights:
                evaluate_updated_model(local_model, X_test, y_test)
            else:
                print("Model henüz güncellenmedi. Lütfen önce modeli güncelleyin.")
        elif choice == "5":
            # Çıkış
            print("Çıkılıyor...")
            break
        else:
            print("Geçersiz seçenek. Lütfen tekrar deneyin.")