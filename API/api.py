#"C:\Users\baki_\AppData\Local\Programs\Python\Python312\Lib\site-packages\pyarc\data_structures\antecedent.py", line 47



from quart import Quart, request, jsonify
import pickle
import base64
import websockets
import asyncio
import time 
import json

app = Quart(__name__)

models = []         # istemcilerden alınan moldeller
global_model = None # birleştirilmiş global model
version = 0         # birleştirilmiş modelin versionu
models_count = 8   # istemcilerden kaç tane model geldiğinde tüm modellerin sunucuya 
                    # gönderileceğini belirleyen değişken. Yani bu değeri 2 yaparsak 
                    # models listesinin eleman sayısı 2 ve daha fazla olduğu durumda 
                    # bu liste içerisindeki modellerin birleştirilmesi için sunucuya 
                    # gönderilir ve models listesi boşaltılır.
sup = 0.2
conf = 0.5
tour = 10
id = 0
select_feature = "mutual"
# feature_selection = {"pca":[],
#                      "chi":['CholCheck', 'Fruits', 'Veggies', 'AnyHealthcare', 'Sex'],
#                      "mutual":['CholCheck', 'Smoker', 'Stroke', 'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'MentHlth', 'Sex','Education']
#                     }
feature_selection = {"pca":[],
                     "chi":['BMI', 'MentalHealth', 'SleepTime'],
                     "mutual":['BMI', 'AlcoholDrinking', 'MentalHealth','Asthma']
                    }
# dataset = "part_"
dataset = "_heart_part_"
# target_col = "Diabetes_binary"
target_col = "HeartDisease"
lock = asyncio.Lock()
tours = []



# modeli sunucuya gönder
# models_count değeri models listesinin eleman sayısından 
# eşit veya fazla ise bu fonksiyon çalışır. 
async def send_models_via_websocket():
    
    try:
        global models
        # gönderilecek veri
        global version
        data = {
                'first': False,
                'models':models
            }
        send_data = pickle.dumps(data)
        models = [] # modeller listesini sil

        # Modeli sunucuya websoket ile gönder
        async with websockets.connect("ws://localhost:7896") as ws:
            await ws.send(send_data)
            print("Modeller Gönderildi")
            await ws.close()

    except Exception as e:
        print(f"WebSocket gönderimi sırasında hata oluştu: {e}")


#Sunucudan Model Al
@app.route('/', methods=['GET'])
async def first_conn():
    global sup
    global conf
    global tour
    global id 
    global dataset
    global target_col
    global models_count
    global select_feature
    global feature_selection
    id = id+1
    dataset_name = str(models_count)+dataset+str(id)+".csv"
    try:
        #modeli gönder
        return jsonify({
            'sup' : sup,
            'conf': conf,
            'tour': tour,
            'id': id,
            'dataset': dataset_name,
            'target_col': target_col,
            'feature_selection': feature_selection[select_feature]
        }),200
        
    except Exception as e:
        print(e)


## İstemciden Model al
@app.route('/send_model', methods=['POST'])
async def get_model_client():
    global models
    global sup
    global conf
    global models_count
    global tour
    global tours
    # JSON verisini al
    data = await request.get_json()
    # İstemcideki modelin version numarası
    version = data['version']
    if not any(model['id'] == data["id"] for model in models):

        # Model verisini hex formatından geri çevir ve pickle ile yükle
        model_data = bytes.fromhex(data['model'])
        model = pickle.loads(model_data)
        print(type(model))
        # Modeli listeye ekle
        async with lock:
            # Models listesinin eleman sayısı models_count değerinden büyükse 
            # bu listeyi sunucuya gönder
            # time.sleep(10)
            models.append({"model":model,'version':data["version"],"size":data['size'],"time":data['time'],'id':data["id"]})
            

        # Eğer models listesinin eleman sayısı yeterliyse sunucuya gönder
        if len(models) >= models_count:
            print("Send Models to Server")
            asyncio.create_task(send_models_via_websocket())
            regular_data = [] 
            for i in models:
                regular_data.append({'size':i['size'],'time':i['time']})
            
            tours.append({
                "tour":version,
                'info':regular_data
            })
            json_data = {
                'sup':sup,
                'conf':conf,
                'tour': tour,
                'count': models_count,
                'tours': tours
            }
            
            # JSON dosyasına yazma işlemi
            with open("models.json", "w", encoding="utf-8") as json_file:
                json.dump(json_data, json_file, ensure_ascii=False, indent=4)



        return f"Model send: {version}", 200
    else:
        return f"Model not send:",204

## İstemciye Model Gönder
@app.route('/get_model', methods=['GET'])
async def send_model_client():
    #İstemcideki modelin version bilgisini al  
    client_model_version = int(request.args.get('version'))
    # global modelin version bilgisi
    global version 


    # Eğer istemci düşük versiondaki bir modeli kullanıyorsa güncel 
    # modeli gönder. Eğer güncel modeli kullanıyorsa gönderme
    if(version > client_model_version):
        global global_model 

        # Modeli uygun hale çevir
        model_data = pickle.dumps(global_model)
        model_base64 = base64.b64encode(model_data).decode('utf-8')

        try:
            #modeli gönder
            return jsonify({
                'model': model_base64,
                'version': version
            }),200
        
        except Exception as e:
            print(e)

    else:
        return  "",204
    

#Sunucudan Model Al
@app.route('/send_federated_model', methods=['POST'])
async def get_federated_model():
    global global_model
    global version

    #sunucudan gelen veriler
    data = await request.get_json()

    #verison bilgisi
    version = data['version']


    # Model verisini hex formatından geri çevir ve pickle ile yükle
    model_data = bytes.fromhex(data['model'])
    model = pickle.loads(model_data)

    # sunucudan gelen modeli global model olarak ayarla
    global_model = model 
    model_path = "model_"+str(version)+".pkl"

        # Modeli kaydet
    model_data = pickle.dumps(model)
    with open(model_path, 'wb') as file:
        file.write(model_data)

    print(f"yeni version: {version}")
    return f"Model Gönderildi", 200


@app.before_serving
async def first_conn():
    try:
        # gönderilecek veri
        data = {
                'first': True
            }
        send_data = pickle.dumps(data)


        # Modeli sunucuya websoket ile gönder
        async with websockets.connect("ws://localhost:7896") as ws:
            await ws.send(send_data)
            
            response_data = await ws.recv()
            response = pickle.loads(response_data)
            print(response)


    except Exception as e:
        print(f"WebSocket gönderimi sırasında hata oluştu: {e}")

async def on_startup():
    await first_conn()



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)