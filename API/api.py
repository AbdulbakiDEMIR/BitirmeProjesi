from quart import Quart, request, jsonify
import pickle
import base64
import websockets
import asyncio
import time
import json

app = Quart(__name__)

# ------------------ #
# GLOBAL DEĞİŞKENLER #
# ------------------ #
models = []             # İstemcilerden gelen modeller
global_model = None     # Birleştirilmiş, güncel model
version = 0             # Modelin mevcut versiyonu
models_count = 2        # Kaç model geldikten sonra sunucuya göndereceğini belirler

# Model eğitimi/feature seçimi ayarları
sup = 0.2
conf = 0.5
tour = 2
id = 0
select_feature = "mutual"
feature_selection = {
    "pca": [],
    "chi": ['BMI', 'MentalHealth', 'SleepTime'],
    "mutual": ['BMI', 'AlcoholDrinking', 'MentalHealth', 'Asthma']
}
dataset = "_heart_part_"
target_col = "HeartDisease"
lock = asyncio.Lock()   # Eşzamanlı erişim için kilit
tours = []              # Tur kayıtları

# ----------------------------------------- #
# Sunucuya WebSocket ile model gönderme fonksiyonu
# ----------------------------------------- #
async def send_models_via_websocket():
    """
    models listesini sunucuya WebSocket üzerinden gönderir.
    Gönderimden sonra models listesi sıfırlanır.
    """
    try:
        global models, version

        data = {
            'first': False,
            'models': models
        }
        send_data = pickle.dumps(data)
        models = []  # Model listesi sıfırlanıyor

        # Modeli WebSocket ile ana sunucuya gönder
        async with websockets.connect("ws://localhost:7896") as ws:
            await ws.send(send_data)
            print("Modeller gönderildi.")
            await ws.close()

    except Exception as e:
        print(f"WebSocket gönderimi sırasında hata oluştu: {e}")

# ---------------------- #
# İlk bağlantıda veri gönderme
# ---------------------- #
@app.route('/', methods=['GET'])
async def first_conn():
    """
    İstemci ilk bağlandığında model eğitim parametrelerini gönderir.
    Her yeni bağlantıda id bir arttırılır ve ona özel bir dataset ismi gönderilir.
    """
    global sup, conf, tour, id, dataset, target_col, models_count, select_feature, feature_selection

    id += 1
    dataset_name = f"{models_count}{dataset}{id}.csv"
    try:
        return jsonify({
            'sup': sup,
            'conf': conf,
            'tour': tour,
            'id': id,
            'dataset': dataset_name,
            'target_col': target_col,
            'feature_selection': feature_selection[select_feature]
        }), 200
    except Exception as e:
        print(e)

# ----------------------------------------- #
# İstemciden Model Alma (POST)
# ----------------------------------------- #
@app.route('/send_model', methods=['POST'])
async def get_model_client():
    """
    İstemciden gelen modeli alır ve models listesine ekler.
    Eğer yeterli sayıda model geldiyse ana sunucuya gönderir.
    """
    global models, sup, conf, models_count, tour, tours

    data = await request.get_json()
    incoming_version = data['version']

    # Aynı id'ye sahip model iki kez eklenmesin
    if not any(model['id'] == data["id"] for model in models):
        # Model verisini pickle ile deserialize et
        model_data = bytes.fromhex(data['model'])
        model = pickle.loads(model_data)
        print("Model türü:", type(model))

        # models listesine ekle
        async with lock:
            models.append({
                "model": model,
                'version': data["version"],
                "size": data['size'],
                "time": data['time'],
                'id': data["id"]
            })

        # Yeterli model geldiyse, sunucuya gönderimi başlat
        if len(models) >= models_count:
            print("Yeterli model geldi, sunucuya gönderiliyor.")
            asyncio.create_task(send_models_via_websocket())

            # Tur bilgilerini kaydet
            regular_data = [{'size': i['size'], 'time': i['time']} for i in models]
            tours.append({
                "tour": incoming_version,
                'info': regular_data
            })
            json_data = {
                'sup': sup,
                'conf': conf,
                'tour': tour,
                'count': models_count,
                'tours': tours
            }
            # JSON'a yaz
            with open("models.json", "w", encoding="utf-8") as json_file:
                json.dump(json_data, json_file, ensure_ascii=False, indent=4)

        return f"Model send: {incoming_version}", 200
    else:
        return "Model not send (duplicate id)", 204

# ----------------------------------------- #
# İstemciye Model Gönderme (GET)
# ----------------------------------------- #
@app.route('/get_model', methods=['GET'])
async def send_model_client():
    """
    İstemciden gelen version parametresiyle karşılaştırıp, gerekiyorsa güncel modeli gönderir.
    """
    client_model_version = int(request.args.get('version'))
    global version, global_model

    # Eğer sunucudaki model daha yeni ise modeli gönder
    if version > client_model_version:
        model_data = pickle.dumps(global_model)
        model_base64 = base64.b64encode(model_data).decode('utf-8')
        try:
            return jsonify({
                'model': model_base64,
                'version': version
            }), 200
        except Exception as e:
            print(e)
    else:
        # Güncel ise model göndermeye gerek yok
        return "", 204

# ----------------------------------------- #
# Ana Sunucudan Federated Model Alma (POST)
# ----------------------------------------- #
@app.route('/send_federated_model', methods=['POST'])
async def get_federated_model():
    """
    Ana sunucudan gelen federated modeli alır ve kaydeder.
    """
    global global_model, version

    data = await request.get_json()
    version = data['version']

    # Modeli yükle ve kaydet
    model_data = bytes.fromhex(data['model'])
    model = pickle.loads(model_data)
    global_model = model
    model_path = f"model_{version}.pkl"
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    print(f"Yeni global model kaydedildi. Versiyon: {version}")
    return "Model Gönderildi", 200

# ----------------------------------------- #
# Sunucu başlatılırken ilk bağlantı ile model çekme
# ----------------------------------------- #
@app.before_serving
async def before_serving():
    """
    Sunucu ilk başlatıldığında ana sunucuya bağlanıp modeli çeker.
    """
    try:
        data = {'first': True}
        send_data = pickle.dumps(data)

        async with websockets.connect("ws://localhost:7896") as ws:
            await ws.send(send_data)
            response_data = await ws.recv()
            response = pickle.loads(response_data)
            print("Başlangıç modeli çekildi:", response)

    except Exception as e:
        print(f"WebSocket ile ilk model çekilirken hata: {e}")

# ----------------------------------------- #
# Sunucuyu başlat
# ----------------------------------------- #
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
