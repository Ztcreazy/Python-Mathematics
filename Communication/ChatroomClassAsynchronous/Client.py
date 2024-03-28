from ChatClassClient import Client
import numpy as np
# import pickle # Create portable serialized representations of Python objects.

if __name__ == "__main__":
    HEADER_LENGTH = 10
    IP = "127.0.0.1"
    PORT = 1234
    my_username = input("Username: ")
    client = Client(IP, PORT, my_username)

    message = input(f'{my_username} > ')
    client.send_message(message)

    while True:
        if client.receive_messages():
            break
        