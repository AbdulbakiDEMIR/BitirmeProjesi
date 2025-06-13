import asyncio
import websockets
import pickle
from ML_class import Server

# Sabit port numarası
PORT = 7896

def log(message):
    """Konsola bilgilendirici mesaj basar."""
    print(f"[SERVER] {message}")

log(f"Sunucu {PORT} portunda dinliyor...")

# Federated model nesnesi oluşturuluyor
federated_model = Server()
federated_model.check_save_model()

async def handle_websocket(websocket):
    """
    Her istemci bağlantısında tetiklenen ana fonksiyon.
    Gelen mesajları alır, modele göre cevap döner veya modeli günceller.
    """
    try:
        # İstemciden mesaj bekle
        recv = await websocket.recv()
        data = pickle.loads(recv)

        # İlk bağlantı mı, yoksa model birleştirme mi?
        is_first_request = data.get("first", False)

        if not is_first_request:
            # Model birleştirme isteği
            models = data.get('models', [])
            log("Model birleştirme isteği alındı.")

            federated_model.fed_avg(models)     # Modelleri birleştir
            federated_model.send_model()        # Yeni modeli gönder
            log("Model birleştirildi ve gönderildi.")

        else:
            # Sunucudaki modeli gönderme isteği
            if federated_model.version > 0:
                log(f"Model (versiyon {federated_model.version}) gönderiliyor.")
                send_data = pickle.dumps({
                    'version': federated_model.version,
                    'model': federated_model.model
                })
                await websocket.send(send_data)
            else:
                log("Henüz bir model yok, versiyon 0 gönderiliyor.")
                send_data = pickle.dumps({'version': 0})
                await websocket.send(send_data)

    except websockets.exceptions.ConnectionClosed as e:
        log("Bir istemci bağlantısı kapandı.")
        log(str(e))
    except Exception as e:
        log(f"Hata oluştu: {e}")

async def main():
    """
    WebSocket sunucusunu başlatır.
    """
    async with websockets.serve(handle_websocket, "localhost", PORT):
        log("WebSocket sunucusu başlatıldı.")
        await asyncio.Future()  # Sunucu sürekli çalışsın

if __name__ == '__main__':
    asyncio.run(main())
