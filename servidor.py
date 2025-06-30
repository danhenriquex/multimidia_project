# servidor.py - Atualizado para usar o CustomAutoencoder treinado

import base64
import json
import os
import random
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit


# ==============================================================================
# 1. DEFINIÇÃO DO NOVO MODELO (copiado de treinar_autoencoder.py)
# ==============================================================================
class CustomAutoencoder(nn.Module):
    """
    Autoencoder customizável baseado na arquitetura fornecida
    Encoder: 784 -> 512 -> 128 -> latent_dim
    Decoder: latent_dim -> 128 -> 512 -> 784
    """

    def __init__(self, input_dim=784, latent_dim=2):
        super(CustomAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded


class MultimediaServer:
    def __init__(self, model_path="./models/best_model.pth"):
        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = "multimedia_server_secret"
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ==============================================================================
        # 2. CARREGAR O MODELO TREINADO
        # ==============================================================================
        self.autoencoder, self.latent_dim, self.model_version = self.load_trained_model(
            model_path
        )
        self.autoencoder.to(self.device)
        self.autoencoder.eval()  # Coloca o modelo em modo de avaliação

        self.is_streaming = False
        self.is_mnist_streaming = False
        self.cap = cv2.VideoCapture(0)

        # ==============================================================================
        # 3. TRANSFORMAÇÃO CONSISTENTE COM O TREINAMENTO
        # ==============================================================================
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # Normaliza para [-1, 1]
            ]
        )
        self.mnist_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=self.transform
        )
        self.mnist_index = 0
        print(f"Dataset MNIST carregado: {len(self.mnist_dataset)} imagens")

        self.setup_routes()
        self.setup_socketio()

    def load_trained_model(self, model_path):
        """Carrega o modelo treinado a partir de um arquivo .pth"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {model_path}")

        print(f"Carregando modelo de {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)

        latent_dim = checkpoint["latent_dim"]
        model = CustomAutoencoder(latent_dim=latent_dim)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Usamos o timestamp do modelo como versão para rastreamento
        version = checkpoint.get("timestamp", datetime.now().isoformat())

        print(f"Modelo carregado com sucesso:")
        print(f"  Dimensão Latente: {latent_dim}")
        print(f"  Versão (Timestamp): {version}")

        return model, latent_dim, version

    def setup_routes(self):
        @self.app.route("/health")
        def health():
            return jsonify(
                {
                    "status": "healthy",
                    "model_version": self.model_version,
                    "latent_dim": self.latent_dim,
                    "device": str(self.device),
                }
            )

        @self.app.route("/model/info")
        def model_info():
            return jsonify(
                {
                    "version": self.model_version,
                    "latent_dim": self.latent_dim,
                    "input_dim": 784,
                    "architecture": "CustomAutoencoder",
                }
            )

        @self.app.route("/model/decoder")
        def get_decoder_weights():
            """Endpoint para clientes baixarem os pesos do decoder"""
            decoder_state = {
                # As chaves já são corretas, sem necessidade de prefixo
                "state_dict": {
                    k: v.cpu().numpy().tolist()
                    for k, v in self.autoencoder.decoder.state_dict().items()
                },
                "version": self.model_version,
                "latent_dim": self.latent_dim,  # Envia latent_dim para o cliente
            }
            return jsonify(decoder_state)

        @self.app.route("/start_stream", methods=["POST"])
        def start_stream():
            if not self.is_streaming:
                self.is_streaming = True
                threading.Thread(target=self.stream_latent_vectors, daemon=True).start()
                return jsonify({"status": "streaming started"})
            return jsonify({"status": "already streaming"})

        @self.app.route("/stop_stream", methods=["POST"])
        def stop_stream():
            self.is_streaming = False
            return jsonify({"status": "streaming stopped"})

        @self.app.route("/start_mnist_stream", methods=["POST"])
        def start_mnist_stream():
            if not self.is_mnist_streaming:
                self.is_mnist_streaming = True
                threading.Thread(target=self.stream_mnist_images, daemon=True).start()
                return jsonify({"status": "MNIST streaming started"})
            return jsonify({"status": "MNIST already streaming"})

        @self.app.route("/stop_mnist_stream", methods=["POST"])
        def stop_mnist_stream():
            self.is_mnist_streaming = False
            return jsonify({"status": "MNIST streaming stopped"})

    def setup_socketio(self):
        @self.socketio.on("connect")
        def handle_connect():
            print(f"Cliente conectado: {request.sid}")
            emit(
                "model_info",
                {"version": self.model_version, "latent_dim": self.latent_dim},
            )

        @self.socketio.on("disconnect")
        def handle_disconnect():
            print(f"Cliente desconectado: {request.sid}")

        @self.socketio.on("request_decoder_update")
        def handle_decoder_request():
            decoder_state = {
                "state_dict": {
                    k: v.cpu().numpy().tolist()
                    for k, v in self.autoencoder.decoder.state_dict().items()
                },
                "version": self.model_version,
                "latent_dim": self.latent_dim,
            }
            emit("decoder_update", decoder_state)

    def preprocess_frame(self, frame):
        """Preprocessa frame da webcam para o formato do autoencoder"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))

        # Aplica a mesma transformação do treinamento
        # Converte para PIL Image para usar as transformações do torchvision
        pil_img = Image.fromarray(resized)
        tensor = self.transform(pil_img)

        # Flatten e envia para o device
        return tensor.view(1, -1).to(self.device)

    def stream_mnist_images(self):
        """Thread para streaming de imagens MNIST"""
        print("Iniciando streaming de imagens MNIST...")
        while self.is_mnist_streaming:
            try:
                # Pega a próxima imagem (já transformada)
                image_tensor, label = self.mnist_dataset[self.mnist_index]
                input_tensor = image_tensor.view(1, -1).to(self.device)

                with torch.no_grad():
                    _, latent_vector = self.autoencoder(input_tensor)

                # Desnormaliza a imagem original para visualização no cliente
                original_img_display = (
                    (image_tensor.squeeze().numpy() * 0.5) + 0.5
                ) * 255
                original_img_display = original_img_display.astype(np.uint8)

                mnist_data = {
                    "latent_vector": latent_vector.cpu().numpy().flatten().tolist(),
                    "original_image": original_img_display.tolist(),
                    "label": int(label),
                    "mnist_index": self.mnist_index,
                    "timestamp": time.time(),
                }
                self.socketio.emit("mnist_data", mnist_data)

                self.mnist_index = (self.mnist_index + 1) % len(self.mnist_dataset)
                time.sleep(5.0)

            except Exception as e:
                print(f"Erro no streaming MNIST: {e}")
                time.sleep(5.0)
        print("Streaming MNIST interrompido")

    def stream_latent_vectors(self):
        """Thread principal de streaming dos vetores latentes"""
        print("Iniciando streaming de vetores latentes...")
        while self.is_streaming:
            try:
                ret, frame = self.cap.read()
                if ret:
                    input_tensor = self.preprocess_frame(frame)

                    with torch.no_grad():
                        _, latent_vector = self.autoencoder(input_tensor)

                    latent_data = {
                        "latent_vector": latent_vector.cpu().numpy().flatten().tolist(),
                        "timestamp": time.time(),
                        "frame_id": int(time.time() * 1000),
                    }
                    self.socketio.emit("latent_vector", latent_data)
                    time.sleep(0.1)
                else:
                    time.sleep(0.5)
            except Exception as e:
                print(f"Erro no streaming: {e}")
                time.sleep(1)
        print("Streaming interrompido")

    def run(self, host="0.0.0.0", port=8080, debug=False):
        print(f"Servidor iniciando em {host}:{port} usando {self.device}")
        self.socketio.run(
            self.app, host=host, port=port, debug=debug, use_reloader=False
        )


if __name__ == "__main__":
    # Caminho para o modelo treinado
    MODEL_FILE = "./models/best_model.pth"
    if not os.path.exists(MODEL_FILE):
        print(
            "ERRO: Modelo treinado 'best_model.pth' não encontrado na pasta './models/'"
        )
        print("Por favor, execute 'treinar_autoencoder.py' primeiro.")
    else:
        server = MultimediaServer(model_path=MODEL_FILE)
        server.run(host="0.0.0.0", port=8080)
