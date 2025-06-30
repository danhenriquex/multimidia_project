# cliente.py - Atualizado para usar o decoder do CustomAutoencoder

import json
import threading
import time
from collections import deque

import cv2
import numpy as np
import requests
import socketio
import torch
import torch.nn as nn


# ==============================================================================
# 1. DEFINIÇÃO DO NOVO DECODER (deve corresponder ao CustomAutoencoder)
# ==============================================================================
class ClientCustomDecoder(nn.Module):
    def __init__(self, latent_dim=2, output_dim=784):
        super(ClientCustomDecoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, z):
        return self.model(z)


class MultimediaClient:
    def __init__(self, server_url="http://localhost:8080"):
        self.server_url = server_url
        self.socket_url = server_url

        self.decoder = None  # Será inicializado após obter latent_dim
        self.latent_dim = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_version = 0
        self.is_connected = False
        self.latest_image = None
        self.latest_mnist_image = None

        self.sio = socketio.Client(reconnection_attempts=5, reconnection_delay=3)
        self.setup_socketio()

        # ... (outras inicializações como stats e buffers) ...
        self.stats = {
            "frames_received": 0,
            "mnist_received": 0,
            "last_frame_time": 0,
            "fps": 0,
            "latency": 0,
            "current_mnist_label": -1,
        }

    def setup_socketio(self):
        @self.sio.event
        def connect():
            print("Conectado ao servidor!")
            self.is_connected = True
            # Solicita a atualização assim que conectar
            self.request_decoder_update()

        @self.sio.event
        def disconnect():
            print("Desconectado do servidor!")
            self.is_connected = False

        @self.sio.event
        def latent_vector(data):
            self.process_latent_vector(data)

        @self.sio.event
        def mnist_data(data):
            self.process_mnist_data(data)

        @self.sio.event
        def decoder_update(data):
            print(f"Recebendo atualização do decoder (versão {data['version']})")
            self.update_decoder(data)

    def connect_to_server(self):
        """Conecta ao servidor via Socket.IO"""
        try:
            print(f"Conectando ao servidor: {self.socket_url}")
            self.sio.connect(self.socket_url)
            return True
        except socketio.exceptions.ConnectionError as e:
            print(f"Erro ao conectar: {e}. O servidor está rodando?")
            return False

    def request_decoder_update(self):
        """Solicita atualização do decoder via Socket.IO"""
        if self.is_connected:
            print("Solicitando pesos do decoder...")
            self.sio.emit("request_decoder_update")
        else:
            print("Não conectado. Tentando baixar via HTTP REST.")
            self.download_decoder_weights()

    def download_decoder_weights(self):
        """Baixa os pesos do decoder via HTTP REST"""
        try:
            response = requests.get(f"{self.server_url}/model/decoder")
            if response.status_code == 200:
                decoder_data = response.json()
                self.update_decoder(decoder_data)
                return True
            return False
        except requests.exceptions.RequestException as e:
            print(f"Erro na requisição HTTP: {e}")
            return False

    def update_decoder(self, decoder_data):
        try:
            new_latent_dim = decoder_data["latent_dim"]

            # Se o decoder não existe ou a dimensão latente mudou, cria um novo
            if self.decoder is None or new_latent_dim != self.latent_dim:
                print(f"Inicializando decoder com latent_dim = {new_latent_dim}")
                self.latent_dim = new_latent_dim
                self.decoder = ClientCustomDecoder(latent_dim=self.latent_dim)
                self.decoder.to(self.device)

            # Carrega os pesos
            state_dict = {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in decoder_data["state_dict"].items()
            }
            self.decoder.model.load_state_dict(state_dict)
            self.decoder.eval()  # Modo de avaliação

            self.model_version = decoder_data["version"]
            print(f"Decoder atualizado com sucesso para a versão: {self.model_version}")

        except Exception as e:
            print(f"Erro ao atualizar decoder: {e}")

    def process_latent_vector(self, data):
        """Processa vetor latente recebido e reconstrói imagem"""
        if self.decoder is None:
            return

        try:
            latent_vector = (
                torch.FloatTensor(data["latent_vector"]).unsqueeze(0).to(self.device)
            )
            with torch.no_grad():
                reconstructed = self.decoder(latent_vector)

            # ==================================================================
            # 2. DESNORMALIZAÇÃO CORRETA DA SAÍDA
            # O modelo retorna valores em [-1, 1]. Convertemos para [0, 255].
            # Fórmula: (valor * 0.5 + 0.5) * 255
            # ==================================================================
            img_tensor = reconstructed.cpu().view(28, 28)
            img_array = ((img_tensor * 0.5) + 0.5).numpy() * 255
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

            display_img = cv2.resize(
                img_array, (280, 280), interpolation=cv2.INTER_NEAREST
            )
            self.latest_image = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

            # ... (Lógica de stats e exibição de texto) ...

        except Exception as e:
            print(f"Erro ao processar vetor latente: {e}")

    def process_mnist_data(self, data):
        """Processa dados MNIST recebidos"""
        if self.decoder is None:
            return

        try:
            latent_vector = (
                torch.FloatTensor(data["latent_vector"]).unsqueeze(0).to(self.device)
            )
            original_image = np.array(data["original_image"], dtype=np.uint8)
            label = data["label"]

            with torch.no_grad():
                reconstructed_tensor = self.decoder(latent_vector)

            # Desnormaliza a imagem reconstruída
            reconstructed_img_tensor = reconstructed_tensor.cpu().view(28, 28)
            reconstructed_img = ((reconstructed_img_tensor * 0.5) + 0.5).numpy() * 255
            reconstructed_img = np.clip(reconstructed_img, 0, 255).astype(np.uint8)

            # Cria a comparação lado a lado
            comparison = np.hstack([original_image, reconstructed_img])
            display_img = cv2.resize(
                comparison, (560, 280), interpolation=cv2.INTER_NEAREST
            )
            self.latest_mnist_image = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

            # ... (Lógica de stats e exibição de texto) ...
            cv2.putText(
                self.latest_mnist_image,
                f"Original | Reconstructed (Label: {label})",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        except Exception as e:
            print(f"Erro ao processar dados MNIST: {e}")

    def start_visualization(self):
        """Inicia visualização em tempo real"""
        print("Iniciando visualização... Pressione 'q' para sair.")
        while True:
            if self.latest_image is not None:
                cv2.imshow("Webcam Stream - Decoded", self.latest_image)

            if self.latest_mnist_image is not None:
                cv2.imshow(
                    "MNIST Stream - Original vs Reconstructed", self.latest_mnist_image
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("u"):
                self.request_decoder_update()
            # ... (outros controles) ...

        cv2.destroyAllWindows()

    def run(self):
        """Executa o cliente"""
        if self.connect_to_server():
            # A thread principal ficará para a visualização
            self.start_visualization()
        else:
            print(
                "Não foi possível conectar ao servidor. Tentando baixar o decoder via HTTP e sair."
            )
            self.download_decoder_weights()
            print("Cliente encerrado.")

        if self.is_connected:
            self.sio.disconnect()


if __name__ == "__main__":
    server_url = input("URL do servidor (padrão: http://localhost:8080): ").strip()
    if not server_url:
        server_url = "http://localhost:8080"

    client = MultimediaClient(server_url)
    client.run()
