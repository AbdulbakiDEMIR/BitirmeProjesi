import os
import json
import pickle
import requests
from pyarc import CBA, TransactionDB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


MODEL_PATH = "model.pkl"
CONFIG_PATH = "load.json"
SERVER_URL = "http://localhost:5000/send_federated_model"


class Server:
    def __init__(self, model_path=MODEL_PATH, config_path=CONFIG_PATH, server_url=SERVER_URL):
        """
        Server objesini başlatır.
        model_path: Modelin kaydedileceği ve yükleneceği dosya yolu.
        config_path: Versiyon ve yol bilgisinin tutulduğu config dosyası.
        server_url: Model gönderilecek sunucu adresi.
        """
        self.model = None
        self.size = 0
        self.version = 0
        self.model_path = model_path
        self.config_path = config_path
        self.server_url = server_url

    def check_save_model(self):
        """
        Sistemde kayıtlı model ve versiyon bilgisini kontrol eder.
        Model varsa yükler, yoksa yeni bir config oluşturur.
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as file:
                config = json.load(file)
            version = config.get("version", 0)
            path = config.get("path", "")
            if version > 0 and os.path.exists(path):
                with open(path, 'rb') as model_file:
                    self.model = pickle.load(model_file)
                self.version = version
                print(f"Mevcut model yüklendi. Versiyon: {self.version}")
            else:
                print("Model versiyonu 0 veya model dosyası yok.")
        else:
            # Config dosyası yoksa, sıfırdan oluştur
            config = {
                "version": self.version,
                "path": ""
            }
            with open(self.config_path, 'w') as file:
                json.dump(config, file, indent=4)
            print("Yeni config dosyası oluşturuldu.")

    def send_model(self):
        """
        Modeli belirtilen sunucuya gönderir.
        Model, pickle ile hex string'e çevrilerek JSON olarak iletilir.
        """
        if self.model is None:
            print("Önce bir model yüklemelisiniz.")
            return

        try:
            # Modeli pickle ile binary formata çevir
            model_data = pickle.dumps(self.model)
            payload = {
                'model': model_data.hex(),   # binary veriyi hex string olarak gönder
                'version': self.version
            }

            print(f"Model gönderiliyor... Versiyon: {self.version}")

            response = requests.post(self.server_url, json=payload)

            if response.status_code == 200:
                print("Başarıyla gönderildi! Sunucu cevabı:", response.text)
            else:
                print(f"Hata oluştu! HTTP Kod: {response.status_code}, Sunucu cevabı: {response.text}")

        except Exception as e:
            print("Model gönderiminde hata oluştu:", str(e))

    def fed_avg(self, models):
        """
        Federated öğrenme için modelleri birleştirir (FedAvg).
        Model yoksa ilk modeli yükler, varsa mevcut modelle diğerlerini birleştirir.
        """
        if not models:
            print("Birleştirilecek model listesi boş.")
            return

        try:
            if self.model is None:
                # Eğer model daha önce yüklenmediyse, ilk model ile başla
                self.model = models[0]["model"]
                self.size = models[0]["size"]
                # update_cba_model2: Model birleştirme fonksiyonunuz
                self.model, self.size = self.model.update_cba_model2(models[1:], self.model, self.size)
            else:
                self.model, self.size = self.model.update_cba_model2(models, self.model, self.size)

            self.version += 1

            # Modeli kaydet
            with open(self.model_path, 'wb') as file:
                pickle.dump(self.model, file)

            # Config dosyasını güncelle
            config = {
                "version": self.version,
                "path": self.model_path
            }
            with open(self.config_path, 'w') as file:
                json.dump(config, file, indent=4)

            print(f"Model birleştirildi ve kaydedildi. Yeni versiyon: {self.version}")

        except Exception as e:
            print("Model birleştirme sırasında hata oluştu:", str(e))

