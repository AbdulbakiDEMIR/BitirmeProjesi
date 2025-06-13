import asyncio
import websockets
import uuid
import pickle
from ML_class import Server


PORT = 7896
print("Server listening on port " + str(PORT))


federe_model = Server()

federe_model.check_save_model()


async def echo(websocket):
    
    try:
        # api mesajı (modeller)
        recv = await websocket.recv()
        data = pickle.loads(recv)
        first = data["first"]
        if(not first):
            models = data['models']

            # modelleri birleştir
            federe_model.fed_avg(models)
            #modeli gönder
            federe_model.send_model()

        else:
            if(federe_model.version > 0):
                data = {
                    'version': federe_model.version,
                    'model':federe_model.model
                }

                send_data = pickle.dumps(data)

                await websocket.send(send_data)
                
            else:
                data = {
                    'version': 0
                }
                send_data = pickle.dumps(data)
                await websocket.send(send_data)

    except websockets.exceptions.ConnectionClosed as e:
        print(f"API disconnected.")
        print(e)

    




async def main():
    # Sunucu görevini ve arka plan görevini aynı anda çalıştırıyoruz.
    async with websockets.serve(echo, "localhost", PORT):
        await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(main())