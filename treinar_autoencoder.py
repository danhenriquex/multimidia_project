# treinar_autoencoder.py - Script para treinar o autoencoder
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


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

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),  # Linear activation (sem ativação)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),  # Linear activation (sem sigmoid para MSE)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # Flatten se necessário
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded


class AutoencoderTrainer:
    def __init__(self, latent_dim=2, learning_rate=0.001, device=None):
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Modelo
        self.model = CustomAutoencoder(latent_dim=latent_dim)
        self.model.to(self.device)

        # Otimizador e loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Histórico de treinamento
        self.history = {"train_loss": [], "val_loss": [], "epochs": []}

        print(f"Autoencoder inicializado:")
        print(f"  Device: {self.device}")
        print(f"  Latent dimension: {latent_dim}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Parâmetros: {sum(p.numel() for p in self.model.parameters()):,}")

    def load_mnist_data(self, batch_size=128, validation_split=0.2):
        """Carrega e prepara dados MNIST"""
        print("Carregando dataset MNIST...")

        # Transformações
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # Normaliza para [-1, 1]
            ]
        )

        # Dataset completo
        full_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )

        # Split treino/validação
        train_size = int((1 - validation_split) * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
        )

        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        # Dataset de teste
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )

        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
        )

        print(f"Dados carregados:")
        print(f"  Treino: {len(train_dataset)} amostras")
        print(f"  Validação: {len(val_dataset)} amostras")
        print(f"  Teste: {len(test_dataset)} amostras")

        return self.train_loader, self.val_loader, self.test_loader

    def train_epoch(self):
        """Treina por uma época"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            reconstructed, encoded = self.model(data)

            # Flatten os dados para calcular loss
            data_flat = data.view(data.size(0), -1)
            loss = self.criterion(reconstructed, data_flat)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Log progresso
            if batch_idx % 100 == 0:
                print(
                    f"    Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.6f}"
                )

        return total_loss / num_batches

    def validate_epoch(self):
        """Valida o modelo"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)

                reconstructed, encoded = self.model(data)
                data_flat = data.view(data.size(0), -1)
                loss = self.criterion(reconstructed, data_flat)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, epochs=30, save_every=5, save_dir="./models"):
        """Treina o autoencoder"""
        print(f"\nIniciando treinamento por {epochs} épocas...")

        # Cria diretório para salvar modelos
        os.makedirs(save_dir, exist_ok=True)

        best_val_loss = float("inf")

        for epoch in range(epochs):
            print(f"\nÉpoca {epoch+1}/{epochs}")

            # Treina
            train_loss = self.train_epoch()

            # Valida
            val_loss = self.validate_epoch()

            # Salva histórico
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["epochs"].append(epoch + 1)

            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")

            # Salva melhor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(os.path.join(save_dir, "best_model.pth"))
                print(f"  ✓ Novo melhor modelo salvo! (Val Loss: {val_loss:.6f})")

            # Salva periodicamente
            if (epoch + 1) % save_every == 0:
                self.save_model(os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))
                print(f"  ✓ Modelo da época {epoch+1} salvo")

        print(f"\nTreinamento concluído!")
        print(f"Melhor Val Loss: {best_val_loss:.6f}")

        # Salva modelo final
        self.save_model(os.path.join(save_dir, "final_model.pth"))

        return self.history

    def save_model(self, filepath):
        """Salva o modelo e metadados"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "latent_dim": self.latent_dim,
                "learning_rate": self.learning_rate,
                "history": self.history,
                "timestamp": datetime.now().isoformat(),
            },
            filepath,
        )

        # Salva configuração em JSON
        config_path = filepath.replace(".pth", "_config.json")
        with open(config_path, "w") as f:
            json.dump(
                {
                    "latent_dim": self.latent_dim,
                    "learning_rate": self.learning_rate,
                    "input_dim": 784,
                    "architecture": {
                        "encoder": [784, 512, 128, self.latent_dim],
                        "decoder": [self.latent_dim, 128, 512, 784],
                    },
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

    def load_model(self, filepath):
        """Carrega modelo salvo"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "history" in checkpoint:
            self.history = checkpoint["history"]

        print(f"Modelo carregado de {filepath}")
        return checkpoint

    def plot_history(self, save_path=None):
        """Plota histórico de treinamento"""
        if not self.history["epochs"]:
            print("Nenhum histórico disponível para plotar")
            return

        plt.figure(figsize=(12, 4))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(
            self.history["epochs"], self.history["train_loss"], "b-", label="Train Loss"
        )
        plt.plot(
            self.history["epochs"], self.history["val_loss"], "r-", label="Val Loss"
        )
        plt.title("Training History")
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)

        # Espaço latente (se 2D)
        plt.subplot(1, 2, 2)
        if self.latent_dim == 2:
            self.plot_latent_space()
        else:
            plt.text(
                0.5,
                0.5,
                f"Latent Space\n{self.latent_dim}D",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title(f"Latent Space ({self.latent_dim}D)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Gráfico salvo em {save_path}")

        plt.show()

    def plot_latent_space(self, num_samples=1000):
        """Plota espaço latente 2D"""
        if self.latent_dim != 2:
            print("Plotagem do espaço latente disponível apenas para latent_dim=2")
            return

        self.model.eval()
        latent_vectors = []
        labels = []

        with torch.no_grad():
            for i, (data, label) in enumerate(self.test_loader):
                if len(latent_vectors) >= num_samples:
                    break

                data = data.to(self.device)
                _, encoded = self.model(data)

                latent_vectors.append(encoded.cpu().numpy())
                labels.append(label.numpy())

        if latent_vectors:
            latent_vectors = np.vstack(latent_vectors)[:num_samples]
            labels = np.hstack(labels)[:num_samples]

            scatter = plt.scatter(
                latent_vectors[:, 0],
                latent_vectors[:, 1],
                c=labels,
                cmap="tab10",
                alpha=0.6,
                s=1,
            )
            plt.colorbar(scatter)
            plt.title("Latent Space Visualization")
            plt.xlabel("Latent Dimension 1")
            plt.ylabel("Latent Dimension 2")

    def test_reconstruction(self, num_examples=8, save_path=None):
        """Testa reconstrução em algumas amostras"""
        self.model.eval()

        # Pega algumas amostras
        dataiter = iter(self.test_loader)
        images, labels = next(dataiter)
        images = images[:num_examples].to(self.device)

        with torch.no_grad():
            reconstructed, encoded = self.model(images)
            reconstructed = reconstructed.view(-1, 28, 28)

        # Visualiza
        fig, axes = plt.subplots(2, num_examples, figsize=(num_examples * 2, 4))

        for i in range(num_examples):
            # Original
            axes[0, i].imshow(images[i].cpu().squeeze(), cmap="gray")
            axes[0, i].set_title(f"Original\nLabel: {labels[i]}")
            axes[0, i].axis("off")

            # Reconstruída
            axes[1, i].imshow(reconstructed[i].cpu(), cmap="gray")
            axes[1, i].set_title("Reconstructed")
            axes[1, i].axis("off")

        plt.suptitle("Original vs Reconstructed Images")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Comparação salva em {save_path}")

        plt.show()


def main():
    """Função principal para treinar o autoencoder"""
    print("=== Treinamento do Autoencoder ===")

    # Configurações
    config = {
        "latent_dim": 2,  # Dimensão do espaço latente (2 como no exemplo)
        "learning_rate": 0.001,  # Taxa de aprendizado
        "batch_size": 128,  # Tamanho do batch
        "epochs": 30,  # Número de épocas
        "validation_split": 0.2,  # Divisão treino/validação
        "save_every": 5,  # Salvar modelo a cada N épocas
    }

    print("Configurações:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Inicializa trainer
    trainer = AutoencoderTrainer(
        latent_dim=config["latent_dim"], learning_rate=config["learning_rate"]
    )

    # Carrega dados
    train_loader, val_loader, test_loader = trainer.load_mnist_data(
        batch_size=config["batch_size"], validation_split=config["validation_split"]
    )

    # Treina
    history = trainer.train(epochs=config["epochs"], save_every=config["save_every"])

    # Plota resultados
    print("\nGerando visualizações...")
    trainer.plot_history(save_path="./models/training_history.png")
    trainer.test_reconstruction(save_path="./models/reconstruction_test.png")

    # Salva histórico
    with open("./models/training_config.json", "w") as f:
        json.dump({**config, "history": history}, f, indent=2)

    print("\n✅ Treinamento concluído!")
    print("Arquivos salvos em ./models/:")
    print("  - best_model.pth (melhor modelo)")
    print("  - final_model.pth (modelo final)")
    print("  - training_history.png (gráficos)")
    print("  - reconstruction_test.png (teste)")
    print("  - training_config.json (configuração)")


if __name__ == "__main__":
    main()
