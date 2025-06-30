#!/usr/bin/env python3
"""
pipeline_completo.py - Demonstração completa do sistema

Este script demonstra todo o pipeline:
1. Treina um autoencoder
2. Integra ao servidor
3. Testa o sistema
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

import requests


class PipelineDemo:
    def __init__(self):
        self.server_url = "http://localhost:8080"
        self.models_dir = "./models"
        self.ensure_directories()

    def ensure_directories(self):
        """Cria diretórios necessários"""
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs("./data", exist_ok=True)
        os.makedirs("./deployment", exist_ok=True)

    def step_1_train_model(self):
        """Passo 1: Treina o autoencoder"""
        print("\n" + "=" * 50)
        print("PASSO 1: Treinando Autoencoder")
        print("=" * 50)

        try:
            # Importa e executa treinamento
            print("Iniciando treinamento...")

            # Simula treinamento rápido para demo
            from treinar_autoencoder import AutoencoderTrainer

            trainer = AutoencoderTrainer(
                latent_dim=8,  # Dimensão maior para melhor performance
                learning_rate=0.001,
            )

            # Carrega dados
            trainer.load_mnist_data(batch_size=256, validation_split=0.2)

            # Treina por menos épocas para demo rápida
            print("Treinando por 5 épocas (demo rápida)...")
            history = trainer.train(epochs=5, save_every=2)

            # Gera visualizações
            trainer.plot_history(save_path=f"{self.models_dir}/training_history.png")
            trainer.test_reconstruction(
                save_path=f"{self.models_dir}/reconstruction_test.png"
            )

            print("✅ Treinamento concluído!")
            return True

        except Exception as e:
            print(f"❌ Erro no treinamento: {e}")
            return False

    def step_2_integrate_model(self):
        """Passo 2: Integra modelo ao servidor"""
        print("\n" + "=" * 50)
        print("PASSO 2: Integrando Modelo ao Servidor")
        print("=" * 50)

        try:
            from integrar_modelo import ModelIntegrator

            integrator = ModelIntegrator()

            # Integra modelo treinado
            trained_model_path = f"{self.models_dir}/best_model.pth"
            server_model_path = f"{self.models_dir}/server_model.pth"

            if not os.path.exists(trained_model_path):
                print(f"❌ Modelo treinado não encontrado: {trained_model_path}")
                return False

            # Atualiza modelo do servidor
            server_path, save_data = integrator.update_server_model(
                trained_model_path, server_model_path
            )

            # Cria pacote de deployment
            integrator.create_deployment_package(server_path)

            print("✅ Integração concluída!")
            print(f"Latent dimension: {save_data['latent_dim']}")
            return True

        except Exception as e:
            print(f"❌ Erro na integração: {e}")
            return False

    def step_3_start_server(self):
        """Passo 3: Inicia servidor (em processo separado)"""
        print("\n" + "=" * 50)
        print("PASSO 3: Iniciando Servidor")
        print("=" * 50)

        print("Iniciando servidor em processo separado...")
        print("Para parar o servidor, pressione Ctrl+C")
        print(f"Servidor estará disponível em: {self.server_url}")

        # Instrução para o usuário
        print("\nEm outro terminal, execute:")
        print("  python servidor.py")
        print("\nPressione Enter quando o servidor estiver rodando...")
        input()

        return self.wait_for_server()

    def wait_for_server(self, max_attempts=30):
        """Aguarda servidor ficar disponível"""
        print("Aguardando servidor...")

        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{self.server_url}/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Servidor disponível!")
                    return True
            except:
                pass

            print(f"Tentativa {attempt + 1}/{max_attempts}...")
            time.sleep(2)

        print("❌ Servidor não respondeu")
        return False

    def step_4_test_system(self):
        """Passo 4: Testa o sistema"""
        print("\n" + "=" * 50)
        print("PASSO 4: Testando Sistema")
        print("=" * 50)

        try:
            # Testa health
            print("1. Testando health check...")
            response = requests.get(f"{self.server_url}/health")
            health_data = response.json()
            print(f"   Status: {health_data['status']}")
            print(f"   Modelo versão: {health_data['model_version']}")
            print(f"   MNIST dataset: {health_data['mnist_dataset_size']} imagens")

            # Testa model info
            print("\n2. Testando informações do modelo...")
            response = requests.get(f"{self.server_url}/model/info")
            model_data = response.json()
            print(f"   Versão: {model_data['version']}")
            print(f"   Latent dim: {model_data['latent_dim']}")
            print(f"   Input dim: {model_data['input_dim']}")

            # Testa MNIST aleatório
            print("\n3. Testando MNIST aleatório...")
            for i in range(3):
                response = requests.get(f"{self.server_url}/mnist/random")
                mnist_data = response.json()
                print(
                    f"   Amostra {i+1}: Label {mnist_data['label']}, Index {mnist_data['index']}"
                )

            # Inicia streaming MNIST
            print("\n4. Iniciando streaming MNIST...")
            response = requests.post(f"{self.server_url}/start_mnist_stream")
            print(f"   {response.json()}")

            print("\n5. Aguardando alguns frames MNIST...")
            time.sleep(15)  # Aguarda 3 frames (5s cada)

            # Para streaming
            print("\n6. Parando streaming...")
            response = requests.post(f"{self.server_url}/stop_mnist_stream")
            print(f"   {response.json()}")

            print("\n✅ Testes concluídos!")
            return True

        except Exception as e:
            print(f"❌ Erro nos testes: {e}")
            return False

    def step_5_start_client(self):
        """Passo 5: Instrução para iniciar cliente"""
        print("\n" + "=" * 50)
        print("PASSO 5: Iniciando Cliente")
        print("=" * 50)

        print("Para ver a visualização completa, execute em outro terminal:")
        print("  python cliente.py")
        print("\nControles do cliente:")
        print("  Q - Sair")
        print("  1 - Iniciar streaming webcam")
        print("  2 - Iniciar streaming MNIST")
        print("  3 - Parar todos streams")
        print("  S - Salvar frames")
        print("  U - Atualizar decoder")

        print("\nPressione Enter para continuar...")
        input()

    def generate_report(self):
        """Gera relatório final"""
        print("\n" + "=" * 50)
        print("RELATÓRIO FINAL")
        print("=" * 50)

        # Verifica arquivos criados
        files_created = []

        check_files = [
            ("Modelo treinado", f"{self.models_dir}/best_model.pth"),
            ("Modelo servidor", f"{self.models_dir}/server_model.pth"),
            ("Histórico treino", f"{self.models_dir}/training_history.png"),
            ("Teste reconstrução", f"{self.models_dir}/reconstruction_test.png"),
            ("Config deployment", "./deployment/DEPLOYMENT.md"),
            ("Dataset MNIST", "./data/MNIST/raw/train-images-idx3-ubyte"),
        ]

        print("Arquivos criados:")
        for name, path in check_files:
            exists = "✅" if os.path.exists(path) else "❌"
            print(f"  {exists} {name}: {path}")
            if os.path.exists(path):
                files_created.append(path)

        # Informações do sistema
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                print(f"\nStatus do servidor:")
                print(f"  ✅ Servidor rodando")
                print(f"  Modelo versão: {health['model_version']}")
                print(f"  Device: {health.get('device', 'N/A')}")
                print(f"  MNIST dataset: {health['mnist_dataset_size']} imagens")
            else:
                print(f"\n❌ Servidor não está respondendo")
        except:
            print(f"\n❌ Servidor não está disponível")

        print(f"\nPipeline concluído!")
        print(f"Total de arquivos criados: {len(files_created)}")

    def run_complete_pipeline(self):
        """Executa pipeline completo"""
        print("🚀 PIPELINE COMPLETO - SISTEMA MULTIMEDIA AUTOENCODER")
        print("Este demo executa todo o pipeline de ponta a ponta")

        start_time = time.time()

        # Pipeline steps
        steps = [
            ("Treinar Autoencoder", self.step_1_train_model),
            ("Integrar Modelo", self.step_2_integrate_model),
            ("Iniciar Servidor", self.step_3_start_server),
            ("Testar Sistema", self.step_4_test_system),
            ("Instruções Cliente", self.step_5_start_client),
        ]

        for step_name, step_func in steps:
            print(f"\n🔄 Executando: {step_name}")
            success = step_func()

            if not success:
                print(f"❌ Pipeline interrompido em: {step_name}")
                return False

        # Relatório final
        self.generate_report()

        end_time = time.time()
        duration = end_time - start_time
        print(f"\n⏱️  Tempo total: {duration:.1f} segundos")

        return True


def main():
    """Função principal"""
    print("Verificando dependências...")

    # Verifica se os scripts necessários existem
    required_files = [
        "treinar_autoencoder.py",
        "integrar_modelo.py",
        "servidor.py",
        "cliente.py",
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("❌ Arquivos necessários não encontrados:")
        for f in missing_files:
            print(f"   - {f}")
        print("\nCertifique-se de ter todos os arquivos do projeto.")
        return

    # Executa pipeline
    demo = PipelineDemo()

    print("\nOpções:")
    print("1. Pipeline completo (treino + integração + teste)")
    print("2. Apenas treinar modelo")
    print("3. Apenas integrar modelo existente")
    print("4. Apenas testar servidor")

    choice = input("\nEscolha uma opção (1-4): ").strip()

    if choice == "1":
        demo.run_complete_pipeline()
    elif choice == "2":
        demo.step_1_train_model()
    elif choice == "3":
        demo.step_2_integrate_model()
    elif choice == "4":
        if demo.wait_for_server():
            demo.step_4_test_system()
    else:
        print("Opção inválida")


if __name__ == "__main__":
    main()
