from Client_class import Client
import time
client = Client(algorithm = "m1")
client.first()
print(client.dataset)
tour = 0
print("id: ",client.id)
while tour != client.tour:


    client.train_model()
    send_model_b = False
    while not send_model_b:
        send_model_b = client.send_model()

    get_model_b = False

    while not get_model_b:
        get_model_b = client.get_model()

    tour = tour+1
    time.sleep(1)
    
client.test_model()