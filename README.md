### Servidor
- ✅ Carrega autoencoder (treinado ou padrão)
- ✅ **Carregamento automático de modelos treinados**
- ✅ Captura entrada# Projeto Multimedia Autoencoder Distribuído

## Visão Geral

Este projeto implementa uma arquitetura distribuída onde:

- **Servidor**: Executa o encoder de um autoencoder e faz streaming dos vetores latentes
- **Clientes**: Recebem os vetores latentes e executam o decoder para reconstruir e visualizar as imagens
- **Treinamento**: Pipeline completo para treinar autoencoders customizados
- **Integração**: Sistema automático para atualizar modelos em produção

## Arquitetura

```
┌─────────────────┐    Vetores Latentes    ┌──────────────────┐
│   PC SERVIDOR   │ ────────────────────► │   PC CLIENTE 1   │
│                 │                        │                  │
│  ┌──────────┐   │   ┌─────────────────┐  │  ┌───────────┐   │
│  │ Encoder  │   │   │   WebSocket/    │  │  │ Decoder   │   │
│  │ (Input)  │   │   │   HTTP API      │  │  │ (Output)  │   │
│  └──────────┘   │   └─────────────────┘  │  └───────────┘   │
│                 │                        │                  │
│  ┌──────────┐   │                        │  ┌───────────┐   │
│  │ Backend  │   │                        │  │ Display   │   │
│  │Streaming │   │                        │  │   GUI     │   │
│  └──────────┘   │                        │  └───────────┘   │
└─────────────────┘                        └──────────────────┘
                                                      │
                                           ┌──────────────────┐
                                           │   PC CLIENTE 2   │
                                           │                  │
                                           │  ┌───────────┐   │
                                           │  │ Decoder   │   │
                                           │  │ (Output)  │   │
                                           │  └───────────┘   │
                                           └──────────────────┘
```

## Instalação

### 1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2. Verificar Instalação do PyTorch

Para GPU (opcional):
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

## Como Executar

### 🚀 Opção 1: Pipeline Completo (Recomendado)

Para demonstração completa do sistema:

```bash
python pipeline_completo.py
```

Este script irá:
1. ✅ Treinar um autoencoder do zero
2. ✅ Integrar o modelo ao servidor  
3. ✅ Testar todas as funcionalidades
4. ✅ Fornecer instruções para clientes

### 🔧 Opção 2: Passo a Passo Manual

#### 1. Treinar Autoencoder (Opcional)

```bash
python treinar_autoencoder.py
```

#### 2. Integrar Modelo Treinado (Opcional)

```bash
python integrar_modelo.py
```

#### 3. Executar o Servidor

No PC que será o servidor:

```bash
python servidor.py
```

O servidor irá:
- ✅ Carregar automaticamente modelos treinados (se disponíveis)
- ✅ Usar modelo padrão se nenhum treinado for encontrado
- ✅ Executar em `http://localhost:8080`

#### 4. Executar os Clientes

Em cada PC cliente:

```bash
python cliente.py
```

Quando solicitado, digite a URL do servidor (ex: `http://192.168.1.100:8080`)

#### 5. Iniciar o Streaming

**Para streaming da webcam:**
```bash
curl -X POST http://localhost:8080/start_stream
```

**Para streaming MNIST (demonstração):**
```bash
curl -X POST http://localhost:8080/start_mnist_stream
```

**Para atualizar modelo:**
```bash
curl -X POST http://localhost:8080/update_model
```

## Endpoints da API

**Gerenciamento:**
- `GET /health` - Status do servidor
- `GET /model/info` - Informações do modelo
- `GET /model/decoder` - Download dos pesos do decoder
- `POST /update_model` - Atualiza modelo treinado

**Streaming:**
- `POST /start_stream` - Inicia streaming da webcam
- `POST /stop_stream` - Para streaming da webcam
- `POST /start_mnist_stream` - Inicia streaming MNIST (5s intervalo)
- `POST /stop_mnist_stream` - Para streaming MNIST

**MNIST:**
- `GET /mnist/random` - Retorna imagem MNIST aleatória

**WebSocket:**
- `/socket.io` - Streaming em tempo real

## Funcionalidades

### Servidor
- ✅ Carrega autoencoder pré-treinado
- ✅ Captura entrada (webcam ou dados sintéticos)
- ✅ **Dataset MNIST integrado para demonstração**
- ✅ Encoda para vetores latentes
- ✅ **Streaming MNIST com timer de 5 segundos**
- ✅ Streaming via WebSocket em tempo real
- ✅ API REST para gerenciamento
- ✅ Atualização automática de modelo
- ✅ Distribuição de decoder para clientes
- ✅ **Endpoint para imagens MNIST aleatórias**

### Cliente
- ✅ Conexão automática ao servidor
- ✅ Download/atualização automática do decoder
- ✅ Reconstrução de imagens em tempo real
- ✅ **Visualização simultânea: webcam e MNIST**
- ✅ **Comparação lado a lado: original vs reconstruída**
- ✅ Visualização com estatísticas (FPS, latência)
- ✅ Reconexão automática
- ✅ **Interface expandida de controle via teclado**
- ✅ **Solicitação de imagens MNIST aleatórias**

## Controles do Cliente

Durante a execução do cliente:

- **Q** - Sair
- **R** - Reconectar ao servidor  
- **U** - Forçar atualização do decoder
- **S** - Salvar frames atuais (webcam + MNIST)
- **M** - Solicitar imagem MNIST aleatória
- **1** - Iniciar streaming da webcam
- **2** - Iniciar streaming MNIST (5s intervalo)
- **3** - Parar todos os streams

## Configurações

### Servidor (`servidor.py`)

```python
# Modificar estas variáveis conforme necessário:
host = '0.0.0.0'          # IP do servidor
port = 8080               # Porta do servidor (mudada de 5000 para 8080)
latent_dim = 64           # Dimensão do espaço latente
input_dim = 784           # Dimensão da entrada (28x28 = 784)
```

### Cliente (`cliente.py`)

```python
# URL do servidor
server_url = 'http://IP_DO_SERVIDOR:8080'
```

## Personalização

### Usar Seu Próprio Autoencoder

Para usar um autoencoder já treinado:

1. **No servidor**, substitua o `SimpleAutoencoder` pela sua arquitetura
2. Carregue os pesos: `self.autoencoder.load_state_dict(torch.load('seu_modelo.pth'))`
3. Ajuste `input_dim` e `latent_dim` conforme seu modelo

### Diferentes Tipos de Entrada

O projeto suporta:
- **Webcam**: Configurado por padrão para streaming em tempo real
- **Dataset MNIST**: Integrado para demonstração e testes
- **Imagens**: Modifique `preprocess_frame()`
- **Dados sintéticos**: Para testes sem webcam
- **Streaming de vídeo**: Substitua `cv2.VideoCapture()`

### Streaming MNIST

O servidor inclui um dataset MNIST completo que pode ser usado para demonstração:

```python
# O servidor automaticamente baixa o MNIST na primeira execução
# Localização: ./data/MNIST/
# 60.000 imagens de treinamento de dígitos 0-9
```

**Funcionalidades MNIST:**
- Streaming automático a cada 5 segundos
- Visualização lado a lado (original vs reconstruída)  
- Labels dos dígitos exibidos
- Buffer de últimas 10 imagens MNIST
- Endpoint para imagens aleatórias

### Protocolo de Comunicação

**Formato do vetor latente:**
```json
{
    "latent_vector": [0.1, -0.5, 0.3, ...],
    "timestamp": 1640995200.123,
    "frame_id": 1234
}
```

**Atualização de modelo:**
```json
{
    "new_version": 2,
    "decoder_state": {
        "state_dict": {...}
    }
}
```

**Dados MNIST:**
```json
{
    "latent_vector": [0.1, -0.5, 0.3, ...],
    "original_image": [[255, 128, ...], ...],
    "label": 7,
    "mnist_index": 1234,
    "timestamp": 1640995200.123,
    "type": "mnist_stream"
}
```

## Solução de Problemas

### Conexão Recusada
- Verifique se o servidor está rodando
- Confirme o IP e porta corretos
- Verifique firewall/antivírus

### Webcam Não Detectada
- O servidor usará dados sintéticos automaticamente
- Para forçar webcam: modifique `cv2.VideoCapture(0)`

### Latência Alta
- Reduza FPS no servidor
- **Para MNIST**: A latência é menor por ser dados locais

### Dataset MNIST Não Carrega
- O servidor criará dados sintéticos automaticamente
- Verifique conexão de internet na primeira execução
- Dataset é salvo em `./data/MNIST/` para uso futuro

### Múltiplas Janelas de Visualização
- **Webcam Stream**: Janela principal para streaming da webcam
- **MNIST Stream**: Janela separada mostrando original vs reconstruída  
- **Statistics**: Janela com estatísticas em tempo real
- Use Alt+Tab para navegar entre janelas