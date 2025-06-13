import pandas as pd
from pyarc import CBA, TransactionDB
import pickle
import random
import requests
import base64
import time

class Client:
    """
    Federated Learning istemcisi.
    Model eğitir, gönderir, alır ve test eder.
    """
    def __init__(self, algorithm):
        self.df = None            # Eğitim verisi DataFrame
        self.algorithm = algorithm
        self.target_col = ""      # Hedef sütun (etiket)
        self.model = None         # Eğitimli model
        self.size = 0             # Veri setinin boyutu
        self.version = 0          # Model versiyonu
        self.time = 0             # Model eğitme süresi
        self.dataset = ""         # Dataset dosya adı/id

    def first(self):
        """
        API'den eğitim parametrelerini ve dataset adını alır,
        veri setini yükler ve özellik seçimi uygular.
        """
        url = 'http://localhost:5000/'  # API url'si

        # HTTP GET isteği ile parametreleri çek
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            print(data)
            self.support = data["sup"]               # minimum support
            self.confidence = data["conf"]           # minimum confidence
            self.tour = data["tour"]                 # eğitim turları
            self.id = data["id"]                     # istemci id'si
            self.dataset = data["dataset"]           # dosya adı
            self.target_col = data["target_col"]     # hedef kolon
            df = pd.read_csv(self.dataset)           # veri setini yükle
            # Özellik seçimi (gereksiz özellikleri drop et)
            self.df = df.drop(columns=data["feature_selection"])

    def train_model(self):
        """
        Modeli eğitim verisiyle eğitir.
        Eğitimde pyarc CBA algoritması kullanılır.
        """
        df = self.df
        size = len(df)
        print(f"Eğitim veri boyutu: {size}")
        self.size = size

        # Verileri TransactionDB formatına çevir
        txns_train = TransactionDB.from_DataFrame(df, target=self.target_col)

        # Modeli başlat
        model = CBA(support=self.support, confidence=self.confidence, algorithm=self.algorithm)

        # Eğitimi başlat ve süreyi ölç
        start_time = time.time()
        model.fit(txns_train)
        end_time = time.time()

        self.model = model
        self.time = end_time - start_time
        print(f"Model eğitildi ({self.time:.2f} sn)")

    def send_model(self):
        """
        Eğitilen modeli API'ye gönderir.
        """
        url = 'http://localhost:5000/send_model'
        # Modeli pickle ile binary yap
        model_data = pickle.dumps(self.model)

        payload = {
            'size': self.size,
            'model': model_data.hex(),    # binary'i hex'e çevir
            'version': self.version,
            'time': self.time,
            'id': self.id
        }

        # POST isteğiyle modeli gönder
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Model gönderildi. Sunucu cevabı:", response.text)
            return True
        else:
            print("Model daha önce gönderildi veya hata oluştu.")
            return False

    def get_model(self):
        """
        API'den en güncel global modeli ister.
        Versiyonu güncelse model güncellenir.
        """
        url = f'http://localhost:5000/get_model?version={self.version}'

        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            version = data["version"]
            model_base64 = data["model"]
            print(f"Yeni modelin versiyonu: {version}")

            self.version = version

            # Modeli base64'ten ve pickle'dan geri çevir
            model_data = base64.b64decode(model_base64)
            self.model = pickle.loads(model_data)

            print("Model başarıyla indirildi.")
            return True
        elif response.status_code == 204:
            print("Zaten güncel model kullanılıyor.")
            return False
        else:
            print(f"Model indirme başarısız oldu. HTTP Durum Kodu: {response.status_code}")
            return False

    def test_model(self):
        """
        Modeli test eder, accuracy ve diğer skorları gösterir.
        """
        # Test için rastgele 300 örnek seç
        data_test = self.df.sample(n=300)
        txns_test = TransactionDB.from_DataFrame(data_test, target=self.target_col)
        print("Test edilen model:", self.model)

        # Doğruluk oranı ve tahminler
        accuracy = self.model.rule_model_accuracy(txns_test)
        predicted = self.model.predict(txns_test)

        # Skor çıktıları
        print("\nConfusion Matrix:\n" +
            str(self.model.rule_model_confusion_matrix(
                pd.Series(txns_test.classes), pd.Series(predicted))))
        print("\nClassification Report:\n" +
            str(self.model.rule_model_classification_report(
                pd.Series(txns_test.classes), pd.Series(predicted))))
        print(f"\nCBA Accuracy of Train: {accuracy:.4f}")

