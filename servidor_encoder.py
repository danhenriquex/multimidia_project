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
from PIL import Image


def stream_mnist_images(self):
    """Thread para streaming de imagens MNIST a cada 5 segundos"""
    print("Iniciando streaming de imagens MNIST (5s intervalo)...")

    while self.is_mnist_streaming:
        try:
            if self.mnist_dataset:
                # Pega próxima imagem do dataset
                image, label = self.get_mnist_image(self.mnist_index)

                # Preprocessa para o autoencoder
                flattened = image.flatten().astype(np.float32) / 255.0
                input_tensor = torch.FloatTensor(flattened).unsqueeze(0).to(self.device)

                # Encode para vetor latente
                with torch.no_grad():
                    latent_vector = self.autoencoder.encode(input_tensor)

                # Prepara dados para envio
                mnist_data = {
                    "latent_vector": latent_vector.cpu().numpy().flatten().tolist(),
                    "original_image": image.tolist(),  # Imagem original para comparação
                    "label": int(label),
                    "mnist_index": self.mnist_index,
                    "timestamp": time.time(),
                    "type": "mnist_stream",
                }

                # Envia para todos os clientes conectados
                self.socketio.emit("mnist_data", mnist_data)

                print(f"MNIST enviado: índice {self.mnist_index}, label {label}")

                # Avança para próxima imagem
                self.mnist_index = (self.mnist_index + 1) % len(self.mnist_dataset)

            else:
                print("Dataset MNIST não disponível")

            # Espera 5 segundos
            time.sleep(5.0)

        except Exception as e:
            print(f"Erro no streaming MNIST: {e}")
            time.sleep(5.0)

    print("Streaming MNIST interrompido")

    def notify_model_update(self):
        """Notifica todos os clientes sobre atualização do modelo"""
        try:
            decoder_state = {
                "state_dict": {
                    k: v.cpu().numpy().tolist()
                    for k, v in self.autoencoder.decoder.state_dict().items()
                },
                "version": self.model_version,
            }

            self.socketio.emit(
                "model_updated",
                {
                    "new_version": self.model_version,
                    "decoder_state": decoder_state,
                    "timestamp": time.time(),
                },
            )

            print(f"Notificação de atualização enviada para todos os clientes")

        except Exception as e:
            print(f"Erro ao notificar atualização: {e}")

        def get_mnist_image(self, index):
            """Obtém uma imagem MNIST pelo índice"""
            try:
                if hasattr(self.mnist_dataset, "__getitem__"):
                    # Dataset PyTorch
                    image, label = self.mnist_dataset[index]
                    if isinstance(image, torch.Tensor):
                        # Converte tensor para numpy array
                        image = image.squeeze().numpy()
                        # Normaliza para [0, 255]
                        image = ((image + 1) * 127.5).astype(np.uint8)
                    else:
                        image = np.array(image)
                else:
                    # Dataset sintético
                    image, label = self.mnist_dataset[index]

                return image, label

            except Exception as e:
                print(f"Erro ao obter imagem MNIST: {e}")
                # Retorna imagem em branco em caso de erro
                return np.zeros((28, 28), dtype=np.uint8), 0

        def create_synthetic_mnist(self):
            """Cria dataset sintético se MNIST não carregar"""
            print("Criando dataset sintético...")
            synthetic_data = []
            for i in range(100):
                # Cria imagem sintética 28x28
                img = np.random.rand(28, 28) * 255
                img = img.astype(np.uint8)
                label = i % 10
                synthetic_data.append((img, label))

            self.mnist_dataset = synthetic_data  # servidor.py - PC Servidor com Encoder e Backend de Streaming


# Autoencoder simples para demonstração
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=64):
        super(SimpleAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)


class MultimediaServer:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = "multimedia_server_secret"
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Carrega ou inicializa o autoencoder
        self.autoencoder = self.load_or_create_autoencoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(self.device)

        # Estado do modelo
        self.model_version = 1
        self.is_streaming = False
        self.is_mnist_streaming = False

        # Configurações de captura (webcam ou dados simulados)
        self.cap = cv2.VideoCapture(0)  # 0 para webcam padrão

        # Carrega dataset MNIST para demonstração
        self.mnist_dataset = None
        self.mnist_index = 0
        self.load_mnist_dataset()

        self.setup_routes()
        self.setup_socketio()

    def load_mnist_dataset(self):
        """Carrega dataset MNIST para demonstração"""
        try:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            )

            # Baixa MNIST se não existir
            self.mnist_dataset = torchvision.datasets.MNIST(
                root="./data", train=True, download=True, transform=transform
            )

            print(f"Dataset MNIST carregado: {len(self.mnist_dataset)} imagens")

        except Exception as e:
            print(f"Erro ao carregar MNIST: {e}")
            # Cria dataset sintético se falhar
            self.create_synthetic_mnist()

    def load_or_create_autoencoder(self):
        """Carrega modelo treinado ou cria um novo"""
        # Lista de possíveis caminhos para modelos treinados
        model_paths = [
            "./models/server_model.pth",
            "./models/best_model.pth",
            "./deployment/autoencoder_model.pth",
            "./models/final_model.pth",
        ]

        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    print(f"Carregando modelo treinado: {model_path}")
                    checkpoint = torch.load(model_path, map_location="cpu")

                    # Extrai configurações
                    latent_dim = checkpoint.get("latent_dim", 64)
                    input_dim = checkpoint.get("input_dim", 784)

                    # Cria modelo com configurações corretas
                    model = SimpleAutoencoder(
                        input_dim=input_dim, latent_dim=latent_dim
                    )

                    # Carrega estado se compatível
                    try:
                        model.load_state_dict(checkpoint["model_state_dict"])
                        print(f"✅ Modelo carregado com sucesso!")
                        print(f"   Latent dimension: {latent_dim}")
                        print(f"   Input dimension: {input_dim}")
                        if "timestamp" in checkpoint:
                            print(f"   Criado em: {checkpoint['timestamp']}")
                        return model
                    except Exception as e:
                        print(f"⚠️  Erro ao carregar estado do modelo: {e}")
                        print("   Criando modelo padrão...")
                        break

                except Exception as e:
                    print(f"⚠️  Erro ao carregar {model_path}: {e}")
                    continue

        # Se nenhum modelo foi carregado, cria um padrão
        print("Criando modelo autoencoder padrão...")
        return SimpleAutoencoder()

    def update_autoencoder_from_file(self, model_path):
        """Atualiza o autoencoder a partir de um arquivo"""
        try:
            print(f"Atualizando modelo de: {model_path}")

            checkpoint = torch.load(model_path, map_location=self.device)
            latent_dim = checkpoint.get("latent_dim", 64)
            input_dim = checkpoint.get("input_dim", 784)

            # Cria novo modelo se dimensões mudaram
            current_latent = self.autoencoder.encoder[-1].out_features
            if current_latent != latent_dim:
                print(f"Criando novo modelo: {current_latent}D -> {latent_dim}D")
                self.autoencoder = SimpleAutoencoder(
                    input_dim=input_dim, latent_dim=latent_dim
                )
                self.autoencoder.to(self.device)

            # Carrega estado
            self.autoencoder.load_state_dict(checkpoint["model_state_dict"])

            # Incrementa versão
            old_version = self.model_version
            self.model_version += 1

            print(f"✅ Modelo atualizado: v{old_version} -> v{self.model_version}")

            # Notifica clientes via WebSocket
            self.notify_model_update()

            return True

        except Exception as e:
            print(f"❌ Erro ao atualizar modelo: {e}")
            return False

    def setup_routes(self):
        @self.app.route("/health")
        def health():
            return jsonify(
                {
                    "status": "healthy",
                    "model_version": self.model_version,
                    "device": str(self.device),
                    "streaming": self.is_streaming,
                    "mnist_streaming": self.is_mnist_streaming,
                    "mnist_dataset_size": (
                        len(self.mnist_dataset) if self.mnist_dataset else 0
                    ),
                }
            )

        @self.app.route("/model/info")
        def model_info():
            return jsonify(
                {
                    "version": self.model_version,
                    "latent_dim": 64,
                    "input_dim": 784,
                    "last_updated": datetime.now().isoformat(),
                }
            )

        @self.app.route("/model/decoder")
        def get_decoder_weights():
            """Endpoint para clientes baixarem os pesos do decoder"""
            decoder_state = {
                "state_dict": {
                    k: v.cpu().numpy().tolist()
                    for k, v in self.autoencoder.decoder.state_dict().items()
                },
                "version": self.model_version,
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

        @self.app.route("/mnist/random")
        def get_random_mnist():
            """Retorna uma imagem MNIST aleatória"""
            try:
                if self.mnist_dataset:
                    idx = random.randint(0, len(self.mnist_dataset) - 1)
                    image, label = self.get_mnist_image(idx)

                    # Converte para base64 para envio via JSON
                    _, buffer = cv2.imencode(".png", image)
                    img_str = base64.b64encode(buffer).decode()

                    return jsonify(
                        {
                            "image": img_str,
                            "label": int(label),
                            "index": idx,
                            "format": "base64_png",
                        }
                    )
                else:
                    return jsonify({"error": "MNIST dataset not available"}), 500
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @self.app.route("/update_model", methods=["POST"])
        def update_model_endpoint():
            """Endpoint para atualizar modelo via upload ou caminho"""
            try:
                data = request.get_json()
                model_path = data.get("model_path") if data else None

                if not model_path:
                    # Tenta caminhos padrão
                    default_paths = [
                        "./models/best_model.pth",
                        "./models/server_model.pth",
                    ]

                    for path in default_paths:
                        if os.path.exists(path):
                            model_path = path
                            break

                if not model_path or not os.path.exists(model_path):
                    return jsonify({"error": "Modelo não encontrado"}), 404

                success = self.update_autoencoder_from_file(model_path)

                if success:
                    return jsonify(
                        {
                            "status": "success",
                            "new_version": self.model_version,
                            "model_path": model_path,
                        }
                    )
                else:
                    return jsonify({"error": "Falha ao atualizar modelo"}), 500

            except Exception as e:
                return jsonify({"error": str(e)}), 500

    def setup_socketio(self):
        @self.socketio.on("connect")
        def handle_connect():
            print(f"Cliente conectado: {request.sid}")
            emit("model_info", {"version": self.model_version, "latent_dim": 64})

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
            }
            emit("decoder_update", decoder_state)

    def preprocess_frame(self, frame):
        """Preprocessa frame da webcam para o formato do autoencoder"""
        # Redimensiona para 28x28 (como MNIST) e converte para grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))

        # Normaliza para [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Flatten para (784,)
        flattened = normalized.flatten()

        return torch.FloatTensor(flattened).unsqueeze(0).to(self.device)

    def stream_latent_vectors(self):
        """Thread principal de streaming dos vetores latentes"""
        print("Iniciando streaming de vetores latentes...")

        while self.is_streaming:
            try:
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        # Preprocessa o frame
                        input_tensor = self.preprocess_frame(frame)

                        # Encode para vetor latente
                        with torch.no_grad():
                            latent_vector = self.autoencoder.encode(input_tensor)

                        # Converte para lista Python para JSON
                        latent_data = {
                            "latent_vector": latent_vector.cpu()
                            .numpy()
                            .flatten()
                            .tolist(),
                            "timestamp": time.time(),
                            "frame_id": int(time.time() * 1000) % 10000,
                        }

                        # Envia para todos os clientes conectados
                        self.socketio.emit("latent_vector", latent_data)

                        print(f"Enviado vetor latente: shape={latent_vector.shape}")
                else:
                    # Se não há webcam, gera dados sintéticos
                    fake_latent = torch.randn(1, 64).to(self.device)
                    latent_data = {
                        "latent_vector": fake_latent.cpu().numpy().flatten().tolist(),
                        "timestamp": time.time(),
                        "frame_id": int(time.time() * 1000) % 10000,
                    }
                    self.socketio.emit("latent_vector", latent_data)
                    print("Enviado vetor latente sintético")

                # Controla FPS (~10 FPS)
                time.sleep(0.1)

            except Exception as e:
                print(f"Erro no streaming: {e}")
                time.sleep(1)

        print("Streaming interrompido")

    def update_model(self, new_model_path=None):
        """Atualiza o modelo e notifica os clientes"""
        try:
            if new_model_path:
                self.autoencoder.load_state_dict(torch.load(new_model_path))

            self.model_version += 1

            # Notifica todos os clientes sobre a atualização
            decoder_state = {
                "state_dict": {
                    k: v.cpu().numpy().tolist()
                    for k, v in self.autoencoder.decoder.state_dict().items()
                },
                "version": self.model_version,
            }

            self.socketio.emit(
                "model_updated",
                {"new_version": self.model_version, "decoder_state": decoder_state},
            )

            print(f"Modelo atualizado para versão {self.model_version}")

        except Exception as e:
            print(f"Erro ao atualizar modelo: {e}")

    def run(self, host="0.0.0.0", port=8080, debug=False):
        print(f"Servidor iniciando em {host}:{port}")
        print(f"Dispositivo: {self.device}")
        print("Endpoints disponíveis:")
        print("  GET  /health - Status do servidor")
        print("  GET  /model/info - Informações do modelo")
        print("  GET  /model/decoder - Download dos pesos do decoder")
        print("  POST /start_stream - Inicia streaming webcam")
        print("  POST /stop_stream - Para streaming webcam")
        print("  POST /start_mnist_stream - Inicia streaming MNIST (5s)")
        print("  POST /stop_mnist_stream - Para streaming MNIST")
        print("  POST /update_model - Atualiza modelo treinado")
        print("  GET  /mnist/random - Retorna imagem MNIST aleatória")
        print("  WebSocket: /socket.io - Streaming em tempo real")

        self.socketio.run(self.app, host=host, port=port, debug=debug)


if __name__ == "__main__":
    server = MultimediaServer()

    # Exemplo de como atualizar o modelo durante execução
    # threading.Timer(30, server.update_model).start()  # Atualiza após 30s

    # Usando porta 8080 por padrão (evita conflitos com porta 5000)
    server.run(host="0.0.0.0", port=8080, debug=True)
