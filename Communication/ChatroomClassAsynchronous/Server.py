from ChatClassServer import Server

if __name__ == "__main__":
    IP = "127.0.0.1"
    PORT = 1234
    server = Server(IP, PORT)
    server.run()