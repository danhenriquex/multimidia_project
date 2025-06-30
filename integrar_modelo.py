# integrar_modelo.py - Script para integrar modelo treinado ao servidor
import json
import os
import shutil
from datetime import datetime

import torch
import torch.nn as nn


class CustomAutoencoder(nn.Module):
    """Mesma arquitetura do script de treinamento"""

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
            nn.Linear(128, latent_dim),
        )

        # Decoder
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


class ModelIntegrator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Integrador inicializado - Device: {self.device}")

    def load_trained_model(self, model_path, config_path=None):
        """Carrega modelo treinado"""
        print(f"Carregando modelo de {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        # Carrega checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extrai configurações do checkpoint ou arquivo de config
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            latent_dim = config.get("latent_dim", 2)
        else:
            latent_dim = checkpoint.get("latent_dim", 2)

        # Cria modelo com configuração correta
        model = CustomAutoencoder(latent_dim=latent_dim)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        print(f"Modelo carregado com sucesso:")
        print(f"  Latent dimension: {latent_dim}")
        print(f"  Parâmetros: {sum(p.numel() for p in model.parameters()):,}")

        return model, checkpoint

    def create_server_compatible_model(self, trained_model, latent_dim):
        """Cria modelo compatível com o servidor"""

        # Classe compatível com o servidor (SimpleAutoencoder adaptado)
        class ServerAutoencoder(nn.Module):
            def __init__(self, input_dim=784, latent_dim=64):
                super(ServerAutoencoder, self).__init__()

                # Encoder - arquitetura personalizada
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Linear(128, latent_dim),
                )

                # Decoder - arquitetura personalizada
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 512),
                    nn.ReLU(),
                    nn.Linear(512, input_dim),
                    nn.Sigmoid(),  # Adiciona sigmoid para compatibilidade
                )

            def encode(self, x):
                return self.encoder(x)

            def decode(self, z):
                return self.decoder(z)

            def forward(self, x):
                z = self.encode(x)
                return self.decode(z)

        # Cria novo modelo
        server_model = ServerAutoencoder(latent_dim=latent_dim)

        # Copia pesos do encoder
        server_model.encoder.load_state_dict(trained_model.encoder.state_dict())

        # Copia pesos do decoder (exceto a última camada que agora tem sigmoid)
        decoder_state = trained_model.decoder.state_dict()
        server_decoder_state = server_model.decoder.state_dict()

        # Copia todas as camadas exceto a última
        for key in decoder_state.keys():
            if "4." not in key:  # Pula a última camada linear
                server_decoder_state[key] = decoder_state[key]

        # Para a última camada, copia os pesos mas ajusta para sigmoid
        server_decoder_state["4.weight"] = decoder_state["4.weight"]
        server_decoder_state["4.bias"] = decoder_state["4.bias"]

        server_model.decoder.load_state_dict(server_decoder_state)

        return server_model

    def save_for_server(self, model, save_path, metadata=None):
        """Salva modelo no formato esperado pelo servidor"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_data = {
            "model_state_dict": model.state_dict(),
            "latent_dim": model.encoder[-1].out_features,
            "input_dim": 784,
            "timestamp": datetime.now().isoformat(),
            "version": 1,
        }

        if metadata:
            save_data.update(metadata)

        torch.save(save_data, save_path)
        print(f"Modelo salvo para servidor: {save_path}")

        return save_data

    def update_server_model(
        self, trained_model_path, server_model_path="./models/server_model.pth"
    ):
        """Pipeline completo para atualizar modelo do servidor"""
        print("=== Atualizando modelo do servidor ===")

        # Carrega modelo treinado
        config_path = trained_model_path.replace(".pth", "_config.json")
        model, checkpoint = self.load_trained_model(trained_model_path, config_path)

        # Extrai dimensão latente
        latent_dim = checkpoint.get("latent_dim", 2)

        # Cria modelo compatível com servidor
        server_model = self.create_server_compatible_model(model, latent_dim)
        server_model.to(self.device)

        # Metadados
        metadata = {
            "training_history": checkpoint.get("history", {}),
            "original_model_path": trained_model_path,
            "conversion_date": datetime.now().isoformat(),
        }

        # Salva modelo do servidor
        save_data = self.save_for_server(server_model, server_model_path, metadata)

        # Testa compatibilidade
        self.test_compatibility(server_model)

        print("✅ Modelo atualizado com sucesso!")
        return server_model_path, save_data

    def test_compatibility(self, model):
        """Testa se o modelo é compatível com o servidor"""
        print("Testando compatibilidade...")

        # Teste com entrada simulada
        test_input = torch.randn(1, 784).to(self.device)

        try:
            with torch.no_grad():
                # Teste encode
                latent = model.encode(test_input)
                print(f"  Encode: OK - Shape: {latent.shape}")

                # Teste decode
                decoded = model.decode(latent)
                print(f"  Decode: OK - Shape: {decoded.shape}")

                # Teste forward
                output = model(test_input)
                print(f"  Forward: OK - Shape: {output.shape}")

                # Verifica ranges
                print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

        except Exception as e:
            print(f"  ❌ Erro de compatibilidade: {e}")
            raise

        print("  ✅ Modelo compatível!")

    def create_deployment_package(self, model_path, output_dir="./deployment"):
        """Cria pacote completo para deployment"""
        print("Criando pacote de deployment...")

        os.makedirs(output_dir, exist_ok=True)

        # Copia modelo
        model_dest = os.path.join(output_dir, "autoencoder_model.pth")
        shutil.copy2(model_path, model_dest)

        # Cria arquivo de instruções
        instructions = """
# Instruções para deployment do modelo

## 1. Atualizar servidor

Substitua o modelo no servidor:
```python
# No servidor.py, no método __init__:
model_path = './deployment/autoencoder_model.pth'
if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
    print(f"Modelo carregado: latent_dim={checkpoint['latent_dim']}")
```

## 2. Reiniciar servidor

```bash
python servidor.py
```

## 3. Clientes receberão automaticamente

Os clientes conectados receberão o decoder atualizado automaticamente.
"""

        with open(os.path.join(output_dir, "DEPLOYMENT.md"), "w") as f:
            f.write(instructions)

        # Cria script de deployment
        deploy_script = f"""#!/usr/bin/env python3
# deploy_model.py - Script para fazer deployment do modelo

import torch
import os
import sys

def deploy_model():
    model_path = '{model_dest}'
    
    if not os.path.exists(model_path):
        print(f"Erro: Modelo não encontrado em {{model_path}}")
        return False
    
    # Carrega e verifica modelo
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print(f"Modelo válido - Latent dim: {{checkpoint['latent_dim']}}")
        
        # Aqui você pode adicionar lógica para atualizar o servidor
        print("Modelo pronto para deployment!")
        return True
        
    except Exception as e:
        print(f"Erro ao carregar modelo: {{e}}")
        return False

if __name__ == "__main__":
    deploy_model()
"""

        with open(os.path.join(output_dir, "deploy_model.py"), "w") as f:
            f.write(deploy_script)

        print(f"Pacote de deployment criado em: {output_dir}")
        print("Arquivos:")
        print(f"  - autoencoder_model.pth")
        print(f"  - DEPLOYMENT.md")
        print(f"  - deploy_model.py")


def main():
    """Função principal"""
    print("=== Integrador de Modelo Treinado ===")

    integrator = ModelIntegrator()

    # Caminhos padrão
    trained_model_path = "./models/best_model.pth"
    server_model_path = "./models/server_model.pth"

    # Verifica se modelo treinado existe
    if not os.path.exists(trained_model_path):
        print(f"❌ Modelo treinado não encontrado: {trained_model_path}")
        print("Execute primeiro: python treinar_autoencoder.py")
        return

    try:
        # Atualiza modelo do servidor
        server_path, save_data = integrator.update_server_model(
            trained_model_path, server_model_path
        )

        # Cria pacote de deployment
        integrator.create_deployment_package(server_path)

        print("\n✅ Integração concluída!")
        print(f"Modelo do servidor salvo em: {server_path}")
        print(f"Latent dimension: {save_data['latent_dim']}")
        print("\nPróximos passos:")
        print("1. Reinicie o servidor: python servidor.py")
        print("2. Os clientes receberão o decoder atualizado automaticamente")
        print("3. Use o deployment package para produção")

    except Exception as e:
        print(f"❌ Erro na integração: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
