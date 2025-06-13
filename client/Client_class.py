import pandas as pd
from pyarc import CBA, TransactionDB
import pickle
import random
import requests
import base64
import time
class Client:
    def __init__(self, algorithm):
        self.df= None # eğitim yapılan veriler
        self.algorithm = algorithm
        self.target_col = ""
        self.model = None         # model
        self.size = 0     
        self.version = 0                         # modelin verisyonu
        self.time = 0
        self.dataset = ""

    def first(self):
        url = f'http://localhost:5000/'  # API url'si

        # HTTP isteğinin gönderilmesi
        response = requests.get(url)

        # Yeni model geldiyse kullanılan model ve versiyon
        # numarası güncellenir ve model yerel olarak kaydedilir
        if response.status_code == 200:

            # JSON yanıtını al
            data = response.json()
            print(data)
            self.support = data["sup"]
            self.confidence = data["conf"]
            self.tour = data["tour"]
            self.id = data["id"]
            self.dataset = data["dataset"]
            self.target_col = data["target_col"]
            df = pd.read_csv(self.dataset)
            self.df = df.drop(columns=data["feature_selection"])

    # model eğitme fonksiyonu 
    def train_model(self,):
        df = self.df
        size = len(df) # her eğitimde test için rastgele sayıda ve rastgele veri seçme
        print(size)
        self.size = size
        # verileri eğitim için hazırlama
        
        
        txns_train = TransactionDB.from_DataFrame(df, target=self.target_col)
        model=CBA(support=self.support, confidence=self.confidence, algorithm=self.algorithm)
        start_time = time.time()
        model.fit(txns_train) # model eğitme
        end_time = time.time()
        self.model = model
        self.time = end_time-start_time 
        # print(self.model.clf.rules)
        # model_data = pickle.dumps(self.model)

    # modeli API'ye gönderme
    def send_model(self):
        url = 'http://localhost:5000/send_model'    # API url'si

        #modelin gönderim formuna getirilmesi
        model_data = pickle.dumps(self.model)

        payload = {
            'size': self.size,
            'model': model_data.hex(),  # binary veriyi JSON içinde hex olarak gönderiyoruz
            'version': self.version,
            'time':self.time,
            'id': self.id
        }
        
        # HTTP isteğinin gönderilmesi
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Response:", response.text)
            return True
        else:
            print("Model daha önce gönderildi")
            return False
    # API'den model talep edlmesi
    def get_model(self):
        url = f'http://localhost:5000/get_model?version={self.version}'  # API url'si

        # HTTP isteğinin gönderilmesi
        response = requests.get(url)

        # Yeni model geldiyse kullanılan model ve versiyon
        # numarası güncellenir ve model yerel olarak kaydedilir
        if response.status_code == 200:

            # JSON yanıtını al
            data = response.json()
            version = data["version"]
            model_base64 = data['model']
            print(f"Yeni modelin versionu: {version}")

            self.version = version

            # Base64 ile kodlanmış veriyi decode et
            model_data = base64.b64decode(model_base64)
            # Pickle ile modeli yükle
            self.model = pickle.loads(model_data)
            
            
            print("Model başarıyla indirildi")
            return True
        elif(response.status_code == 204):
            print("Model Güncel")
            return False
        else:
            print(f"Model indirme başarısız oldu. HTTP Durum Kodu: {response.status_code}")
            return False
    # Modeli test et
    def test_model(self):
        # Test için rastgele veri seç
        size = random.randint(300,2000)
        data_test = self.df.sample(n=300)

        # Verileri test için hazır hale getir
        txns_test = TransactionDB.from_DataFrame(data_test, target=self.target_col)
        print(self.model)
        # Testi gerçekleştir
        accuracy = self.model.rule_model_accuracy(txns_test)
        predicted = self.model.predict(txns_test)
        print("\n\n")
        print("Confusion Matrix" + " :\n" +
            str(self.model.rule_model_confusion_matrix(pd.Series(txns_test.classes), pd.Series(predicted))))
        print("Classification Report " + " :\n"+ str(self.model.rule_model_classification_report(pd.Series(txns_test.classes), pd.Series(predicted))))
        print("CBA Accuracy of Train " + ": " + str(accuracy))