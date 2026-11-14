# GTA6 Clipping Machine

**Escreva o que aconteceu e receba o clipe. Uma IA que assiste ao seu gameplay por você, encontra
os momentos que você descreve em palavras simples e os transforma em clipes verticais para
TikTok, Instagram Reels e YouTube Shorts. Movida pelo modelo de IA open source do Google, rodando
inteiramente no seu computador.**

[![CI](https://github.com/RodrigoFlorencio86/GTA6-Clipping-Machine/actions/workflows/ci.yml/badge.svg)](https://github.com/RodrigoFlorencio86/GTA6-Clipping-Machine/actions/workflows/ci.yml)
[![Modelo de IA](https://img.shields.io/badge/IA-Google%20EmbeddingGemma%202-4285F4)](https://huggingface.co/google/embeddinggemma-2)
![Roda localmente](https://img.shields.io/badge/roda-100%25%20local-orange)
![Sem chave de API](https://img.shields.io/badge/chave%20de%20API-n%C3%A3o%20precisa-brightgreen)
[![Licença: MIT](https://img.shields.io/badge/licen%C3%A7a-MIT%20(livre%2C%20com%20cr%C3%A9dito)-green)](#licença-livre-para-usar-com-crédito)

🇺🇸 [Read in English](README.md)

Você gravou três horas de gameplay. Em algum lugar ali tem uma perseguição policial que terminou
em explosão, uma queda ridícula e um drift perfeito. Achar esses momentos significa assistir ao
arquivo inteiro. Em vez disso, digite **"perseguição policial com explosão"** e receba uma lista
de momentos ordenada, com timeline, previews e score. Ajuste o corte, aprove e exporte um MP4
1080x1920 pronto para postar.

- **Sem upload.** Seu gameplay nunca sai do seu computador.
- **Sem assinatura, sem chave de API, sem custo por minuto.** A IA roda na sua máquina, de graça.
- **Sem precisar de placa de vídeo.** Roda na CPU de um notebook comum; uma GPU NVIDIA só deixa
  mais rápido.
- **Busca no seu idioma.** O modelo entende mais de 100 idiomas: "perseguição policial com
  explosão" acha o mesmo momento que a busca em inglês.
- **Open source de ponta a ponta.** O app é MIT; o modelo de IA é Apache-2.0.

> Feito para GTA, funciona com qualquer jogo. Sem afiliação com a Rockstar Games ou a Take-Two Interactive.

## A IA por trás: EmbeddingGemma 2, open source do Google

O motor de busca é o [**EmbeddingGemma 2**](https://huggingface.co/google/embeddinggemma-2), um
modelo de IA multimodal open source lançado pelo Google sob a licença Apache-2.0, da mesma
família dos modelos Gemma. Tem cerca de 744 milhões de parâmetros, pequeno o bastante para rodar
num notebook, e é um dos poucos modelos abertos que entende **texto, imagem, vídeo e áudio num
único "espaço de significado"**.

Na prática:

1. Cada janela de 8 segundos da sua gravação (quatro quadros mais o áudio) vira uma lista de 768
   números, um *embedding*, que resume o que está acontecendo: uma perseguição à noite, uma festa
   na cobertura, um tiroteio num beco.
2. A sua busca também vira 768 números, no mesmo espaço.
3. As janelas mais próximas da sua busca são os momentos que você procura. Sem palavras-chave,
   sem marcar nada à mão, sem decorar minutagem: a IA compara **significado**.

O modelo é baixado uma vez do Hugging Face (cerca de 1,5 GB) e depois roda offline na sua
máquina. Ele só serve para *entender* o vídeo: não gera conteúdo, e nada é enviado ao Google nem
a ninguém.

| | Serviço típico de cortes na nuvem | GTA6 Clipping Machine |
|---|---|---|
| Seu gameplay | enviado para o servidor de alguém | fica no seu disco |
| Custo | plano mensal ou créditos por minuto | grátis |
| Conta / chave de API | obrigatória | nenhuma |
| Funciona offline | não | sim, depois de baixar o modelo uma vez |
| Código | fechado | aberto (MIT), cada etapa pode ser conferida |
| Publicação | alguns postam nas suas contas por você | nunca: você aprova e posta |

Por baixo, tudo é open source e local: **EmbeddingGemma 2** (entendimento), **FFmpeg**
(decodificação, recorte e codificação), **PySceneDetect** (cortes de cena), **LanceDB** (busca
vetorial), **SQLite** (estado e histórico) e **Streamlit** (a interface web em `127.0.0.1`).

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
  central ou quadro inteiro sobre fundo desfocado. Faixas pretas gravadas em volta da imagem
  (jogos 4:3, cutscenes com letterbox, canvas do OBS) são removidas automaticamente. Thumbnail
  automática. Legendas queimadas opcionais, com transcrição local.
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

Na CPU, indexar leva cerca de **1 a 1,5 vez a duração do vídeo**, uma vez por gravação
(medido num notebook Ryzen 7 de 8 núcleos sem placa de vídeo dedicada: gameplay de 9 minutos em
1080p indexado em 11,5 minutos). Depois disso, cada busca leva **cerca de 0,3 segundo**, quantas
vezes você quiser. Uma GPU CUDA indexa bem mais rápido e é usada automaticamente.

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
  Um resultado precisa passar de 0,65 *e* ficar a até 0,04 do melhor resultado da busca (a
  *janela de relevância*, nas configurações de busca); aumente a janela se vierem poucos
  resultados.
- Áudio alto pesa 20% no score. A janela de relevância impede que uma cena barulhenta e pouco
  parecida passe na frente da certa, mas entre momentos de relevância parecida o mais
  barulhento costuma vir primeiro. Ajuste os pesos em `processing.yaml`.
- Tags e presença de reação são heurísticas.
- O recorte inteligente usa uma posição horizontal por clipe e se desloca pouco: o centro da
  tela, onde jogos em terceira pessoa mantêm o personagem ou o carro, sempre fica no quadro.
  Na mira por cima do ombro o personagem fica à esquerda do centro, fora de um corte 9:16;
  nesses clipes, use o enquadramento com fundo desfocado.
- Indexar na CPU é lento para sessões longas.

## Licença: livre para usar, com crédito

O GTA6 Clipping Machine é software livre sob a [licença MIT](LICENSE), criado por
**Rodrigo Florencio** ([@RodrigoFlorencio86](https://github.com/RodrigoFlorencio86)).

- ✅ **Use de graça**, para fins pessoais ou comerciais: no seu canal, para clientes, num estúdio.
- ✅ **Modifique, faça fork, construa em cima e redistribua**, inclusive dentro de outros produtos.
- 📝 **A única condição: dar crédito ao autor.** Qualquer cópia, fork ou produto que inclua este
  código, ou uma parte substancial dele, precisa manter o aviso de copyright
  `Copyright (c) 2026 Rodrigo Florencio` e o texto da licença.
- 🎬 **Os clipes que você faz são seus.** Sem marca d'água, e nenhum crédito é exigido dentro dos
  seus vídeos. Se a ferramenta te poupar tempo, uma menção é sempre bem-vinda:

  ```text
  Cortado com o GTA6 Clipping Machine, de Rodrigo Florencio
  https://github.com/RodrigoFlorencio86/GTA6-Clipping-Machine
  ```

O EmbeddingGemma 2 é distribuído pelo Google sob a licença Apache-2.0; FFmpeg, PyTorch, LanceDB,
Streamlit e PySceneDetect têm suas próprias licenças.

Se o projeto for útil para você, **deixe uma ⭐ no GitHub**: isso ajuda outros criadores a
encontrá-lo.

*Grand Theft Auto e GTA são marcas da Take-Two Interactive Software, Inc. Este projeto é
independente e não tem afiliação, endosso ou patrocínio da Rockstar Games, Take-Two Interactive,
Google, TikTok, Instagram ou YouTube.*
