# GTA6 Clipping Machine

**Encontre os melhores momentos do seu próprio gameplay descrevendo-os em palavras e transforme
esses momentos em clipes verticais para TikTok, Instagram Reels e YouTube Shorts. Tudo roda no
seu computador.**

🇺🇸 [Read in English](README.md)

Você gravou três horas de gameplay. Em algum lugar ali tem uma perseguição policial que terminou
em explosão, uma queda ridícula e um drift perfeito. Em vez de assistir ao arquivo inteiro,
digite **"perseguição policial com explosão"** e receba uma lista de momentos ordenada, com
timeline, previews e score. Depois ajuste o corte, aprove e exporte um MP4 1080x1920.

O projeto usa o [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), modelo
aberto de embeddings multimodais do Google DeepMind: texto, frames de vídeo e áudio ficam no
mesmo espaço vetorial, então suas palavras são comparadas diretamente com o que aparece na tela
e no som. A busca funciona em mais de 100 idiomas.

> Feito para GTA, funciona com qualquer jogo. Sem afiliação com a Rockstar Games ou a Take-Two Interactive.

## Recursos

- **Busca em linguagem natural**: "drift em curva fechada", "queda engraçada", "tiroteio
  intenso com fuga", "pôr do sol cinematográfico".
- **Momentos, não frames**: resultados próximos viram intervalos com tamanho de clipe (8 a 60 s,
  mirando 15 a 45 s).
- **Score editorial**: correspondência semântica, intensidade de áudio, movimento, presença de
  reação, novidade em relação aos clipes já feitos e adequação de duração. Cada score vem com a
  composição e o motivo em linguagem simples.
- **Timeline** dos momentos sobre o vídeo inteiro, com curvas de áudio e movimento.
- **Ajuste manual** de início e fim, com novo cálculo do score.
- **Exportação vertical** 1080x1920 H.264/AAC com *recorte inteligente* (segue a ação), recorte
  central ou quadro inteiro sobre fundo desfocado. Thumbnail automática. Legendas queimadas
  opcionais, com transcrição local.
- **Fila de aprovação**: todo clipe espera sua decisão. O app não publica nada.
- **Privacidade**: só arquivos locais (sem download por link), processamento local, sem
  telemetria, funciona offline depois do primeiro download do modelo.
- Interface em português e inglês. CLI para automação.

## Requisitos

| | Mínimo | Recomendado |
|---|---|---|
| Sistema | Windows 10/11, macOS, Linux | |
| Python | 3.11 | 3.12 ou 3.13 |
| FFmpeg | qualquer versão recente no `PATH` | 6.0+ |
| RAM | 16 GB | 32 GB |
| GPU | não obrigatória (funciona na CPU) | NVIDIA com 8 GB+ de VRAM |

Na CPU, indexar leva cerca de **1 a 1,5 vez a duração do vídeo** (medido: vídeo de 96 s em
113 s num notebook Ryzen 7 de 8 núcleos). A busca é rápida depois que o modelo carrega (~0,2 s).
Uma GPU CUDA indexa bem mais rápido e é usada automaticamente.

## Instalação rápida

```bash
# 1. FFmpeg
winget install Gyan.FFmpeg        # Windows
brew install ffmpeg               # macOS
sudo apt install ffmpeg           # Debian / Ubuntu

# 2. App
git clone https://github.com/RodrigoFlorencio86/GTA6-Clipping-Machine.git
cd GTA6-Clipping-Machine
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -e .                  # com GPU NVIDIA: instale antes o PyTorch CUDA (pytorch.org)

# 3. Verificar e baixar o modelo (uma vez, ~1,5 GB)
clipping-machine doctor
clipping-machine download-model

# 4. Abrir
clipping-machine run              # http://127.0.0.1:8501
```

Para usar em português, escolha o idioma na barra lateral ou defina `GCM_LANG=pt`. Depois do
download do modelo, `GCM_OFFLINE=1` bloqueia qualquer acesso à rede.

Docker (opcional): `docker compose up --build app`. Coloque as gravações em `./videos` e importe
pelo caminho `/videos/arquivo.mp4`.

## Como usar

1. **Vídeos**: confirme que o gameplay é seu e arraste um arquivo (MP4, MOV, MKV, WEBM; até
   4 GB por upload) ou digite o caminho de um arquivo no disco. Importar pelo caminho indexa
   gravações enormes **no lugar**, sem copiar. A indexação começa sozinha e mostra o progresso.
2. **Buscar momentos**: descreva o que procura. Ajuste similaridade mínima, quantidade de
   segmentos e intervalo de agrupamento em *Configurações de busca*. Filtre por score e duração.
3. Em cada momento: veja **score**, **motivo** e **composição**, assista ao **preview**, ajuste
   **início/fim** e clique em **Gerar clipe**.
4. **Fila de aprovação**: assista ao rascunho, ajuste, **aprove** ou **rejeite**. Clipes aprovados
   podem ser **exportados** com o enquadramento escolhido. Baixe o MP4 ou encontre-o em
   `data/clips/`.
5. **Status e privacidade**: estado do modelo e do FFmpeg e métricas locais.

CLI:

```bash
clipping-machine index "D:\Gravacoes\sessao-01.mkv" --i-own-this-footage
clipping-machine search "perseguição policial com explosão"
```

## Configuração

Tudo fica em [`configs/`](configs/): `processing.yaml` (janelas, busca, pesos do score,
exportação), `models.yaml` (EmbeddingGemma 2) e `tags.yaml` (conceitos usados nos motivos;
descreva o *seu* gameplay, em qualquer idioma). Os padrões seguem a especificação: janela de
8 s, overlap de 2 s, keyframe a cada 2 s, embedding de 768 dimensões, similaridade mínima 0,65,
agrupamento abaixo de 3 s, clipes de 15 a 45 s e no máximo 60 s.

Detalhes: [arquitetura](docs/ARCHITECTURE.md) · [modelos e medições](docs/MODELS.md) ·
[roadmap](docs/ROADMAP.md).

## Uso aceitável

Use **apenas com gameplay gravado por você** a partir de uma cópia legítima do jogo, ou com
material que você tem autorização para usar. O app não baixa vídeos por link, de propósito, e
nunca publica nada por você. Você é responsável pelos direitos do que importa e publica, e por
seguir a política de conteúdo da publicadora do jogo e as regras de cada plataforma. Texto
completo (em inglês): [docs/ACCEPTABLE_USE.md](docs/ACCEPTABLE_USE.md).

O score ajuda a revisar primeiro os momentos mais promissores. Ele **não** prevê visualizações
nem viralização.

## Limitações

- Busca semântica não é detecção de eventos: eventos muito curtos ou específicos podem escapar.
  Reescrever a consulta ou baixar a similaridade mínima costuma ajudar.
- As similaridades ficam próximas (sem relação ≈ 0,5–0,6; correspondência clara ≈ 0,7–0,78).
- Áudio alto pesa 20% no score e pode colocar um momento barulhento acima de uma correspondência
  semântica um pouco melhor. Ajuste os pesos em `processing.yaml`.
- Tags e presença de reação são heurísticas.
- O recorte inteligente usa uma posição horizontal por clipe.
- Indexar na CPU é lento para sessões longas.

## Licença

[MIT](LICENSE). *Grand Theft Auto e GTA são marcas da Take-Two Interactive Software, Inc. Este
projeto é independente e não tem afiliação, endosso ou patrocínio da Rockstar Games, Take-Two
Interactive, Google, TikTok, Instagram ou YouTube.*
