# GTA6 Clipping Machine — Especificação de Produto e Implementação

## 1. Visão do produto

**GTA6 Clipping Machine** é uma aplicação local, focada em privacidade, que transforma vídeos de gameplay enviados pelo usuário em clipes curtos para redes sociais.

O usuário faz upload de um arquivo de vídeo — por exemplo, `.mp4`, `.mkv` ou `.mov` — e pesquisa por descrições em linguagem natural, como:

- “perseguição policial com explosão”
- “momento engraçado com queda”
- “drift em curva fechada”
- “tiroteio intenso com fuga”
- “cena cinematográfica sem spoiler”
- “melhores 10 momentos de ação”

O sistema indexa o vídeo localmente usando **EmbeddingGemma 2**, encontra intervalos relevantes, agrupa trechos próximos, atribui um score editorial, permite revisão humana e exporta clipes verticais com FFmpeg.

**Importante:** o produto deve processar apenas vídeos enviados pelo próprio usuário. Não haverá download automático de vídeos do YouTube, TikTok, Twitch ou outras plataformas.

## 2. Objetivos

### Objetivos de produto

- Permitir que um criador encontre momentos específicos em vídeos longos de gameplay sem assistir ao vídeo inteiro.
- Executar indexação e busca de forma local, sem enviar o vídeo ou embeddings para serviços de terceiros.
- Gerar clipes verticais prontos para revisão e publicação.
- Manter controle humano antes de qualquer publicação.
- Ser extensível para outros jogos, canais e formatos de conteúdo.

### Não objetivos do MVP

- Baixar vídeos a partir de links do YouTube, TikTok, Twitch ou outras plataformas.
- Publicar automaticamente sem aprovação do usuário.
- Detectar ou incentivar uso de cheats, exploits ou conteúdo ilegal.
- Garantir viralização ou citação por motores de IA.
- Fazer edição cinematográfica totalmente automática sem revisão humana.

## 3. Público-alvo

- Criadores de conteúdo de games.
- Editores de clipes e shorts.
- Agências e operações de social media.
- Streamers que desejam reaproveitar gravações longas.
- Usuários que valorizam processamento local e privacidade.

## 4. Experiência do usuário

### Fluxo principal

1. O usuário abre a aplicação local.
2. Ele faz upload de um vídeo de gameplay.
3. O sistema valida formato, tamanho, duração e resolução.
4. O usuário inicia a indexação.
5. O sistema extrai frames, chunks de áudio e metadados temporais.
6. O sistema gera embeddings com EmbeddingGemma 2.
7. O sistema armazena os vetores em um banco vetorial local.
8. O usuário faz uma consulta em linguagem natural.
9. O sistema retorna uma lista de momentos candidatos.
10. O usuário revisa, ajusta início/fim e seleciona clipes.
11. O sistema exporta MP4 vertical com legendas opcionais.
12. O usuário decide publicar manualmente ou por integração oficial.

### Tela inicial

A tela inicial deve conter:

- Área de upload por drag-and-drop.
- Lista de vídeos já indexados.
- Barra de progresso de indexação.
- Campo de busca semântica.
- Filtros por vídeo, duração, score e status.
- Timeline interativa.
- Botão “Gerar clipe”.
- Botão “Exportar MP4”.
- Fila de aprovação.

### Exemplo de consulta

```text
Consulta: perseguição policial com explosão

Resultados:
1. 12:04–12:19 — score 0,93 — perseguição, tiros e explosão
2. 27:41–27:58 — score 0,88 — fuga de polícia e carro em chamas
3. 35:10–35:26 — score 0,85 — perseguição em motocicleta
```

## 5. Funcionalidades do MVP

### 5.1 Upload de vídeo

O usuário deve poder enviar arquivos:

- `.mp4`
- `.mov`
- `.mkv`
- `.webm`

Requisitos:

- Aceitar arquivos grandes, mas limitar o tamanho por configuração.
- Validar se o arquivo é um vídeo legível.
- Extrair duração, resolução, frame rate, codec e tamanho.
- Gerar um `video_id` único.
- Armazenar metadados em SQLite.
- Nunca enviar o arquivo original para serviços externos.

### 5.2 Indexação local

O sistema deve processar o vídeo em duas dimensões:

- **Visual:** frames e cenas.
- **Áudio:** chunks de áudio e, opcionalmente, transcrição.

Cada segmento deve receber:

```json
{
  "video_id": "gameplay_001",
  "segment_id": "seg_000123",
  "start_ms": 184000,
  "end_ms": 197500,
  "duration_ms": 13500,
  "embedding": [],
  "embedding_model": "google/embeddinggemma-2",
  "embedding_dimension": 768,
  "preview_frame_path": "data/previews/gameplay_001/seg_000123.jpg",
  "visual_description": "Perseguição em carro à noite com explosão",
  "audio_energy": 0.81,
  "transcript": "",
  "status": "indexed"
}
```

### 5.3 Busca semântica

O usuário deve poder buscar por linguagem natural.

Exemplos:

- “perseguição com explosão”
- “carro pegando fogo”
- “drift em curva”
- “queda engraçada”
- “tiroteio intenso”
- “fuga da polícia”
- “momento cinematográfico”

O sistema deve:

- Gerar o embedding da consulta.
- Buscar por similaridade cosseno.
- Retornar os melhores segmentos.
- Agrupar resultados temporalmente próximos.
- Retornar intervalos, não frames isolados.
- Mostrar preview, score, motivo e timestamp.

### 5.4 Agrupamento de momentos

Resultados muito próximos devem ser agrupados em um único momento.

Regras iniciais:

- Se dois resultados relevantes estiverem a menos de 3 segundos de distância, agrupar.
- Se o intervalo final for menor que 8 segundos, expandir para no mínimo 8 segundos.
- Se o intervalo final for maior que 60 segundos, dividir em candidatos menores.
- Duração alvo inicial: 15 a 45 segundos.
- Permitir ajuste manual de início e fim.

### 5.5 Score editorial

O score não deve depender apenas da similaridade semântica.

Fórmula inicial:

```text
score =
  0.35 * similaridade_semantica
+ 0.20 * intensidade_audio
+ 0.15 * movimento_visual
+ 0.15 * presenca_de_reacao
+ 0.10 * novidade_em_relacao_a_clipes_anteriores
+ 0.05 * adequacao_de_duracao
```

Sinais possíveis:

- Picos de áudio.
- Mudança brusca de cena.
- Movimento intenso de câmera.
- Presença de explosões, tiros, colisões ou perseguições.
- Reação do criador, se houver webcam.
- Repetição de conteúdo já clipado.
- Duração adequada para Shorts, Reels e TikTok.

### 5.6 Geração de clipes

O sistema deve permitir:

- Recorte por intervalo.
- Conversão para 9:16.
- Recorte inteligente para manter o centro da ação, quando possível.
- Legendas queimadas opcionais.
- Thumbnail automática.
- Exportação em MP4 H.264.
- Exportação em resolução 1080x1920 ou configuração equivalente.
- Nome de arquivo padronizado.

Exemplo:

```text
gta6_clip_2026-11-14_12-04-19_score093.mp4
```

### 5.7 Fila de aprovação

Todo clipe deve passar por revisão humana antes de publicação.

Estados possíveis:

- `candidate`
- `approved`
- `rejected`
- `exported`
- `published`
- `archived`

O MVP deve exportar o clipe, mas não publicar automaticamente.

## 6. EmbeddingGemma 2

### 6.1 O que é

**EmbeddingGemma 2** é um modelo de embeddings multimodal do Google DeepMind, projetado para busca semântica on-device.

Ele mapeia texto, código, imagens, vídeo e áudio — inclusive combinações dessas modalidades — para um espaço vetorial compartilhado de 768 dimensões. [web:88][web:89][web:91]

### 6.2 Especificações relevantes

| Característica | Valor |
|---|---|
| Parâmetros totais | 740M [web:73][web:82] |
| Backbone de texto | 270M [web:73][web:82] |
| Encoder de visão | 170M [web:73][web:82] |
| Encoder de áudio | 300M [web:73][web:82] |
| Dimensão do embedding | 768 [web:73][web:82] |
| Contexto | 8.192 tokens [web:73][web:82] |
| Licença | Apache-2.0 [web:73][web:82] |
| Modalidades | Texto, código, imagem, vídeo e áudio [web:76][web:82] |
| Uso indicado | Busca semântica, RAG, classificação, clustering e recuperação multimodal [web:73][web:82] |

### 6.3 Papel no projeto

O EmbeddingGemma 2 é o **motor de busca semântica**.

Ele deve ser usado para:

- Indexar frames, cenas, chunks de áudio e descrições.
- Indexar consultas do usuário.
- Encontrar momentos semanticamente semelhantes.
- Suportar busca multimodal, por exemplo: “encontre cenas parecidas com esta imagem”.

Ele **não deve** ser tratado como:

- Um modelo de edição de vídeo.
- Um modelo de transcrição.
- Um modelo de geração de texto.
- Um detector automático de momentos virais.
- Um sistema que garante que um clipe será publicado ou viral.

### 6.4 Estratégia de indexação

Para cada vídeo:

1. Extrair keyframes.
2. Dividir o vídeo em janelas temporais.
3. Extrair áudio por janela.
4. Gerar embedding por janela.
5. Armazenar metadados e embedding no banco vetorial.
6. Permitir busca por texto, imagem, áudio ou vídeo.

Configuração inicial sugerida:

```yaml
video:
  sample_fps: 1
  keyframe_interval_seconds: 2
  window_duration_seconds: 8
  window_overlap_seconds: 2
  audio_sample_rate: 16000
  max_video_duration_minutes: 180

embedding:
  model: google/embeddinggemma-2
  dimension: 768
  reduced_dimensions: [128, 256, 512]
  normalize: true

search:
  top_k: 50
  min_similarity: 0.65
  merge_gap_seconds: 3
  min_clip_duration_seconds: 8
  max_clip_duration_seconds: 60
  target_clip_duration_seconds: [15, 45]
```

## 7. Como funciona o Video Moment Finder

O **Video Moment Finder** é uma funcionalidade do Google AI Edge Gallery que permite busca semântica multimodal em vídeos, executada 100% no dispositivo. [web:106][web:107][web:108]

### 7.1 Pipeline original

O fluxo do recurso original é:

1. O usuário seleciona ou importa um vídeo.
2. O sistema divide o vídeo em janelas temporais.
3. Extrai keyframes e trechos de áudio.
4. Gera embeddings multimodais com EmbeddingGemma 2.
5. Armazena os vetores em um índice local.
6. O usuário faz uma consulta em linguagem natural.
7. O sistema gera o embedding da consulta.
8. O sistema realiza busca ANN no índice vetorial.
9. Retorna timestamps relevantes.
10. Agrupa intervalos próximos.
11. Permite pular para o momento e recortar/exportar. [web:102][web:104][web:111]

### 7.2 Implementação de referência

O repositório oficial é:

- **GitHub:** https://github.com/google-ai-edge/gallery
- **Licença:** Apache-2.0
- **Plataforma principal:** Android
- **Recurso:** Video Moment Finder
- **Modelo:** EmbeddingGemma 2

O arquivo relevante é:

```text
gallery/Android/src/app/src/main/java/com/google/ai/edge/gallery/customtasks/videomomentfinder/VideoMomentFinderViewModel.kt
```

Esse arquivo demonstra:

- Indexação por janelas temporais.
- Parâmetros como `framesPerWindow` e `windowDurationSec`.
- Busca no vector store.
- Filtro por projeto.
- Extração de frames de prévia.
- Fusão de intervalos sobrepostos.
- Criação de clipes e thumbnails. [web:111]

### 7.3 O que copiar e o que adaptar

| Elemento | Decisão |
|---|---|
| Conceito de janelas temporais | Copiar |
| Busca vetorial multimodal | Copiar |
| Agrupamento de intervalos | Copiar |
| Preview e timeline | Adaptar para desktop/web |
| Indexação local | Copiar |
| Download por link | Não implementar |
| Publicação automática | Não implementar no MVP |
| Score editorial | Criar specifically para gameplay |
| Interface de revisão | Criar specifically para clipes |

## 8. Stack técnica recomendada

### 8.1 Backend

- **Python 3.11+**
- **FastAPI** para API local opcional.
- **Streamlit** para interface rápida do MVP.
- **FFmpeg** para processamento de vídeo e áudio.
- **OpenCV** para extração de frames.
- **PySceneDetect** para detecção de cenas.
- **faster-whisper** para transcrição local opcional.
- **LanceDB** ou **Qdrant local** para busca vetorial.
- **SQLite** para metadados e estado do pipeline.
- **Celery/RQ** opcional para filas locais, se necessário.

### 8.2 Modelos

| Função | Modelo/ferramenta | Status no MVP |
|---|---|---|
| Embeddings multimodais | EmbeddingGemma 2 | Obrigatório |
| Transcrição | faster-whisper | Opcional |
| Descrição de cenas | Qwen2.5-VL ou SmolVLM | Recomendado |
| Detecção de cenas | PySceneDetect | Obrigatório |
| Corte e exportação | FFmpeg | Obrigatório |
| Busca vetorial | LanceDB ou Qdrant | Obrigatório |

### 8.3 Hardware mínimo sugerido

| Perfil | Hardware | Experiência esperada |
|---|---|---|
| Mínimo | CPU moderna, 16 GB RAM | Indexação lenta, vídeos curtos |
| Recomendado | GPU com 8 GB VRAM, 32 GB RAM | Fluxo fluido para vídeos médios |
| Ideal | GPU com 12–24 GB VRAM, 32–64 GB RAM | Indexação e busca mais rápidas |

## 9. Estrutura do repositório

```text
gta6-clipping-machine/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .env.example
├── docker-compose.yml
├── Makefile
├── docs/
│   ├── ARCHITECTURE.md
│   ├── ACCEPTABLE_USE.md
│   ├── MODELS.md
│   └── ROADMAP.md
├── src/
│   ├── ingest/
│   │   ├── video_importer.py
│   │   ├── video_validator.py
│   │   ├── scene_splitter.py
│   │   ├── frame_extractor.py
│   │   └── audio_chunker.py
│   ├── analysis/
│   │   ├── transcriber.py
│   │   ├── scene_describer.py
│   │   ├── audio_analyzer.py
│   │   └── highlight_scorer.py
│   ├── embeddings/
│   │   ├── embeddinggemma_client.py
│   │   ├── text_embedder.py
│   │   ├── image_embedder.py
│   │   ├── audio_embedder.py
│   │   └── video_embedder.py
│   ├── search/
│   │   ├── vector_store.py
│   │   ├── query_engine.py
│   │   ├── moment_finder.py
│   │   └── interval_merger.py
│   ├── clipping/
│   │   ├── clip_builder.py
│   │   ├── vertical_formatter.py
│   │   ├── subtitle_burner.py
│   │   └── thumbnail_generator.py
│   ├── publishing/
│   │   ├── approval_queue.py
│   │   └── youtube_publisher.py
│   ├── storage/
│   │   ├── database.py
│   │   ├── models.py
│   │   └── repositories.py
│   └── ui/
│       ├── streamlit_app.py
│       └── components/
├── prompts/
│   ├── scene_description.txt
│   ├── highlight_rules.txt
│   └── social_caption.txt
├── configs/
│   ├── models.yaml
│   ├── processing.yaml
│   └── channels.yaml
├── data/
│   ├── uploads/
│   ├── processed/
│   ├── index/
│   ├── previews/
│   └── clips/
└── tests/
```

## 10. Contratos de API

### 10.1 Upload de vídeo

```http
POST /api/videos/upload
Content-Type: multipart/form-data
```

Resposta:

```json
{
  "video_id": "gameplay_001",
  "filename": "gameplay.mp4",
  "duration_seconds": 2520,
  "status": "uploaded"
}
```

### 10.2 Indexar vídeo

```http
POST /api/videos/{video_id}/index
```

Resposta:

```json
{
  "video_id": "gameplay_001",
  "status": "indexing",
  "segments_created": 0
}
```

### 10.3 Buscar momentos

```http
POST /api/search
Content-Type: application/json
```

Corpo:

```json
{
  "video_id": "gameplay_001",
  "query": "perseguição policial com explosão",
  "top_k": 20,
  "min_score": 0.65,
  "merge_gap_seconds": 3,
  "target_duration": [15, 45]
}
```

Resposta:

```json
{
  "results": [
    {
      "start_ms": 724000,
      "end_ms": 739000,
      "score": 0.93,
      "reason": "Perseguição, tiros e explosão",
      "preview_frame": "/previews/gameplay_001/seg_000123.jpg"
    }
  ]
}
```

### 10.4 Gerar clipe

```http
POST /api/clips
Content-Type: application/json
```

Corpo:

```json
{
  "video_id": "gameplay_001",
  "start_ms": 724000,
  "end_ms": 739000,
  "format": "9:16",
  "add_subtitles": true,
  "title": "Perseguição insana com explosão"
}
```

## 11. Regras de qualidade

### 11.1 Critérios de aceitação do MVP

O sistema será considerado funcional quando permitir:

- Fazer upload de um vídeo local.
- Indexar o vídeo localmente.
- Buscar por descrição em linguagem natural.
- Retornar pelo menos 10 momentos candidatos.
- Agrupar resultados próximos.
- Mostrar preview, timestamp e score.
- Permitir ajuste manual de início e fim.
- Exportar um clipe MP4 9:16.
- Armazenar histórico de clipes.
- Impedir publicação automática sem aprovação.

### 11.2 Métricas

- Tempo de indexação por minuto de vídeo.
- Tempo de resposta da busca.
- Precisão percebida pelo usuário.
- Taxa de clipes aprovados.
- Taxa de clipes rejeitados.
- Tempo médio para encontrar um momento relevante.
- Número de clipes exportados por vídeo.

## 12. Privacidade e segurança

- Todo processamento deve ocorrer localmente.
- O vídeo original não deve ser enviado a serviços externos.
- Embeddings e metadados devem permanecer no dispositivo ou em armazenamento controlado pelo usuário.
- O usuário deve poder apagar vídeo, índice, embeddings e clipes.
- Não coletar telemetria por padrão.
- Não incluir chaves de API no repositório.
- Usar `.env` para configurações sensíveis.
- Não publicar automaticamente sem aprovação explícita.

## 13. Direitos autorais e uso aceitável

O projeto deve ser usado apenas com **gameplay próprio**, gravado pelo usuário a partir de uma cópia legítima do jogo.

A política da Rockstar Games/Take-Two indica que, em geral, a empresa não se opõe ao uso de materiais por fãs para fins **não comerciais**, desde que não haja vazamento, spoiler isolado ou incentivo a cheats. A política também reserva o direito de remover conteúdo e distingue esse uso de exploração comercial da propriedade intelectual. [web:95]

Regras obrigatórias:

- Não aceitar conteúdo pirata, vazado ou obtido de forma não autorizada.
- Não aceitar gameplay de terceiros sem autorização.
- Não usar a ferramenta para baixar conteúdo de plataformas.
- Não publicar automaticamente sem aprovação.
- Não incentivar cheats, exploits ou violação dos termos do jogo.
- Não prometer monetização, alcance ou viralização.
- Incluir aviso de que o usuário é responsável pelos direitos do conteúdo enviado.

## 14. Roadmap

### Fase 1 — MVP de busca

- Upload local.
- Extração de frames e áudio.
- Embeddings com EmbeddingGemma 2.
- Índice vetorial local.
- Busca por texto.
- Timeline e resultados.
- Exportação de clipes.

### Fase 2 — Curadoria inteligente

- Score editorial.
- Detecção de picos de áudio.
- Detecção de movimento.
- Descrição visual com VLM local.
- Agrupamento mais inteligente.
- Deduplicação de clipes semelhantes.

### Fase 3 — Produção social

- Legendas automáticas.
- Templates de capa.
- Formatos para TikTok, Reels e Shorts.
- Fila de aprovação.
- Publicação manual assistida.

### Fase 4 — Escala

- Processamento em lote.
- Suporte a múltiplos canais.
- Biblioteca de clipes.
- Busca por imagem de referência.
- Busca por áudio de referência.
- Relatórios de performance.

## 15. Referências oficiais

### EmbeddingGemma 2

- Página oficial do Google DeepMind:  
  https://deepmind.google/models/gemma/embeddinggemma/

- Documentação oficial do Google AI:  
  https://ai.google.dev/gemma/docs/embeddinggemma

- Model card do EmbeddingGemma 2:  
  https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2

- Anúncio oficial do Google:  
  https://blog.google/innovation-and-ai/technology/developers-tools/embeddinggemma-2/

- Hugging Face — modelo oficial:  
  https://huggingface.co/google/embeddinggemma-2

- Hugging Face — EmbeddingGemma 300M:  
  https://huggingface.co/google/embeddinggemma-300m

- Hugging Face — documentação Transformers:  
  https://huggingface.co/docs/transformers/model_doc/embedding_gemma2

### Google AI Edge Gallery e Video Moment Finder

- Repositório oficial:  
  https://github.com/google-ai-edge/gallery

- README oficial:  
  https://github.com/google-ai-edge/gallery/blob/main/README.md

- Release com EmbeddingGemma 2:  
  https://github.com/google-ai-edge/gallery/releases/tag/1.0.20

- Implementação do Video Moment Finder:  
  https://github.com/google-ai-edge/gallery/blob/8f5f2a0bdb7c81693384dbdecbabec9b1e8d7f4e/Android/src/app/src/main/java/com/google/ai/edge/gallery/customtasks/videomomentfinder/VideoMomentFinderViewModel.kt

- Blog oficial sobre EmbeddingGemma 2 no AI Edge:  
  https://developers.googleblog.com/google-ai-edge-with-embeddinggemma-2/

### MediaPipe

- Universal Embedder:  
  https://developers.google.com/edge/mediapipe/solutions/retrieval/universal_embedder

- MediaPipe Tasks para agentes de IA:  
  https://developers.google.com/edge/mediapipe/solutions/agents

### Ferramentas open source

- FFmpeg:  
  https://ffmpeg.org/

- OpenCV:  
  https://opencv.org/

- PySceneDetect:  
  https://github.com/Breakthrough/PySceneDetect

- faster-whisper:  
  https://github.com/SYSTRAN/faster-whisper

- LanceDB:  
  https://lancedb.com/

- Qdrant:  
  https://qdrant.tech/

- Streamlit:  
  https://streamlit.io/

- FastAPI:  
  https://fastapi.tiangolo.com/

### Políticas e direitos

- Política da Rockstar Games sobre material protegido por direitos autorais:  
  https://support.rockstargames.com/articles/7bNaeoMFTV0iUDGhStTXvz/policy-on-posting-copyrighted-Rockstar-Games-material

- Termos do YouTube Data API:  
  https://developers.google.com/youtube/terms/api-services-terms-of-service

- Políticas para desenvolvedores do YouTube:  
  https://developers.google.com/youtube/terms/developer-policies

## 16. Prompt inicial para Claude Code

```text
Crie um repositório chamado GTA6 Clipping Machine.

Objetivo:
Construir uma aplicação local em Python que permita ao usuário fazer upload de vídeos de gameplay próprios, indexá-los localmente com EmbeddingGemma 2, buscar momentos por linguagem natural e gerar clipes verticais para revisão e exportação.

Regras críticas:
- Não implemente download por link do YouTube, TikTok, Twitch ou outras plataformas.
- Aceite apenas upload local de arquivos de vídeo.
- Todo processamento deve ocorrer localmente.
- Não envie vídeo, áudio, frames ou embeddings para serviços externos.
- Não publique automaticamente sem aprovação humana.
- Use EmbeddingGemma 2 apenas como motor de embeddings e busca semântica.
- Use FFmpeg para processamento e exportação de vídeo.
- Use LanceDB ou Qdrant local para busca vetorial.
- Use SQLite para metadados e estado do pipeline.
- Use Streamlit para a interface do MVP.
- Use Python 3.11+.
- Escreva código modular, testável e com tipagem.
- Crie testes unitários para extração, indexação, busca, agrupamento e exportação.
- Crie documentação de instalação, uso, arquitetura e limitações.

Pipeline obrigatório:
1. Upload local do vídeo.
2. Validação do arquivo.
3. Extração de metadados.
4. Detecção de cenas.
5. Extração de keyframes.
6. Extração de chunks de áudio.
7. Geração de embeddings multimodais com EmbeddingGemma 2.
8. Armazenamento em índice vetorial local.
9. Busca semântica por consulta em texto.
10. Agrupamento de resultados próximos em intervalos.
11. Score editorial de destaque.
12. Visualização em timeline.
13. Ajuste manual de início e fim.
14. Exportação de clipe MP4 9:16.
15. Fila de aprovação.

Configuração inicial:
- Janela de 8 segundos.
- Overlap de 2 segundos.
- Keyframe a cada 2 segundos.
- Embedding de 768 dimensões.
- Similaridade mínima de 0,65.
- Agrupar resultados com intervalo menor que 3 segundos.
- Duração alvo de clipe: 15 a 45 segundos.
- Duração máxima: 60 segundos.

Score editorial:
score =
  0.35 * similaridade_semantica
+ 0.20 * intensidade_audio
+ 0.15 * movimento_visual
+ 0.15 * presenca_de_reacao
+ 0.10 * novidade_em_relacao_a_clipes_anteriores
+ 0.05 * adequacao_de_duracao

Interface mínima:
- Upload por drag-and-drop.
- Lista de vídeos indexados.
- Barra de progresso.
- Campo de busca.
- Timeline com momentos.
- Preview de cada momento.
- Score e motivo.
- Botão para gerar clipe.
- Botão para exportar MP4.
- Fila de aprovação.

Não implemente:
- Download por URL.
- Publicação automática.
- Coleta de telemetria.
- Upload para nuvem.
- Uso de conteúdo de terceiros sem autorização.
- Garantias de viralização.

Entregue:
1. Estrutura completa do repositório.
2. Código funcional do MVP.
3. README detalhado.
4. Requisitos e instruções de instalação.
5. Configuração do EmbeddingGemma 2.
6. Testes.
7. Docker Compose opcional.
8. Documentação de arquitetura.
9. Documentação de uso aceitável.
10. Roadmap de próximas fases.
```
