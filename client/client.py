from Client_class import Client
import time

# ---- İstemci Nesnesini Oluştur ----
# Algoritma tipi (ör: "m1") ile Client nesnesi başlatılıyor
client = Client(algorithm="m1")

# ---- Sunucudan Eğitim Parametrelerini ve Veriyi Al ----
client.first()
print("Kullanılan veri seti:", client.dataset)
tour = 0  # Eğitim turu (epoch)
print("İstemci ID:", client.id)

# ---- Eğitim Döngüsü (Tur Sayısı Kadar) ----
while tour != client.tour:

    # 1. Modeli eğit
    client.train_model()

    # 2. Model gönderimi (başarılı olana kadar dener)
    send_model_b = False
    while not send_model_b:
        send_model_b = client.send_model()

    # 3. Güncel modeli sunucudan al (başarılı olana kadar dener)
    get_model_b = False
    while not get_model_b:
        get_model_b = client.get_model()

    # 4. Sonraki tura geç ve bir süre bekle
    tour += 1
    time.sleep(1)  # Sunucunun işleme yetişmesi için küçük bir bekleme

# ---- Eğitim Sonrası Test ----
client.test_model()
