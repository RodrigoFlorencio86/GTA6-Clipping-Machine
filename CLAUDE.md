# CLAUDE.md — GTA6 Clipping Machine

> **Agente instalando o app para um usuário final?** Siga o [AGENTS.md](AGENTS.md) e ignore o
> resto deste arquivo: as seções abaixo são notas do mantenedor (ambiente, decisões, testes).

Instruções para o Claude Code trabalhar neste repositório. Leia inteiro antes de mudar qualquer coisa.
A especificação de produto original é [GTA6-Clipping-Machine-SPEC.md](GTA6-Clipping-Machine-SPEC.md) (pt-BR).

## O projeto

Aplicação **local-first** em Python 3.11+: o usuário faz upload do **próprio** gameplay, o app indexa
o vídeo com **EmbeddingGemma 2** (`google/embeddinggemma-2`, 768d, texto/imagem/vídeo/áudio no
mesmo espaço vetorial), busca momentos por linguagem natural, agrupa resultados em intervalos,
calcula um score editorial e exporta clipes **MP4 9:16** (1080x1920, H.264) após **aprovação humana**.

Objetivo do dono: um repositório open source útil para criadores do mundo todo (TikTok, Reels,
Shorts) e que ganhe estrelas no GitHub. Por isso: código, UI e docs principais em **inglês**,
com UI e README também em **pt-BR**.

## Regras inegociáveis (vêm da spec — nunca relaxe)

- **Sem download por URL** (YouTube, TikTok, Twitch etc.). Só arquivos locais. `video_validator.looks_like_url` rejeita links.
- **Tudo local.** Nenhum vídeo, áudio, frame ou embedding sai da máquina. O único acesso à rede é baixar os pesos do modelo uma vez; `GCM_OFFLINE=1` bloqueia até isso.
- **Sem telemetria.** `privacy.enforce_local_only()` desliga a do HF Hub e a do Streamlit; `.streamlit/config.toml` tem `gatherUsageStats = false` e `address = 127.0.0.1`.
- **Sem publicação automática.** Não existe módulo publisher. O status `published` é só registro manual do usuário. A exportação final exige o clipe em `approved` (`ApprovalQueue.require_approved`).
- EmbeddingGemma 2 é usado **só** para embeddings/busca (não gera texto, não transcreve, não decide).
- Stack obrigatória: FFmpeg (processamento/exportação), LanceDB (vetores), SQLite (metadados/estado), Streamlit (UI).
- Sem garantias de viralização em textos, UI ou docs. Aviso de que o usuário é responsável pelos direitos do conteúdo. Sem afiliação com Rockstar/Take-Two.
- `tests/test_guardrails.py` impõe parte disso (imports proibidos, deps proibidas, sem publisher). Não enfraqueça esses testes.

## Comandos

Ambiente: Windows 11. Venv em `.venv` (Python 3.13), gerenciado com `uv`.

```bash
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"   # instalar (editável)
.venv/Scripts/python.exe -m pytest                  # testes (pula os marcados `model`)
.venv/Scripts/python.exe -m pytest -m model         # teste opt-in com o modelo real (~1,5 GB)
.venv/Scripts/python.exe -m ruff check src tests    # lint
.venv/Scripts/python.exe -m ruff format src tests   # formatação
.venv/Scripts/python.exe -m mypy                    # tipos (strict; ui/ excluída)
.venv/Scripts/clipping-machine.exe run              # UI Streamlit em http://127.0.0.1:8501
.venv/Scripts/clipping-machine.exe doctor           # checa ffmpeg, torch, cache do modelo
.venv/Scripts/clipping-machine.exe download-model   # baixa/valida os pesos
.venv/Scripts/clipping-machine.exe index <arquivo> --i-own-this-footage
.venv/Scripts/clipping-machine.exe search "consulta" [--sort score|similarity|time]
```

Variáveis de ambiente (todas opcionais, ver `.env.example`): `GCM_DATA_DIR`, `GCM_CONFIG_DIR`,
`GCM_EMBEDDING_MODEL`, `GCM_DEVICE`, `GCM_OFFLINE`, `GCM_LANG`, `GCM_FFMPEG`, `GCM_FFPROBE`.

Marcadores pytest: `ffmpeg` (pulado se não houver ffmpeg), `model` (excluído por padrão via `addopts`).
Os testes nunca carregam o modelo real: usam `tests/fakes.py::FakeEmbedder` (um eixo por conceito).

## Arquitetura (src/clipping_machine)

| Pacote | Papel |
|---|---|
| `config.py` | Pydantic (`extra="forbid"`, frozen) lendo `configs/processing.yaml`, `models.yaml`, `tags.yaml` + env |
| `privacy.py` | desliga telemetria; modo offline |
| `ffmpeg.py` | wrappers tipados de ffmpeg/ffprobe (sempre lista de args, nunca shell); `probe`, `run_ffmpeg` com progresso, `iter_gray_frames` |
| `ingest/` | `video_validator` (URL, extensão, tamanho, ffprobe, duração, resolução) · `video_importer` (upload ou caminho local in-place; exige confirmação de direitos; fingerprint anti-duplicata) · `windows` (janelas 8 s / overlap 2 s; grade de keyframes 2 s) · `frame_extractor` (keyframes JPEG + movimento num único decode) · `audio_chunker` (WAV mono 16 kHz) · `scene_splitter` (PySceneDetect) |
| `analysis/` | `audio_analyzer` (dBFS, normalização por vídeo) · `scene_describer` (tags zero-shot e sinal de reação via embeddings de texto) · `highlight_scorer` (fórmula do score, `duration_fit`, `novelty`) · `transcriber` (faster-whisper opcional, só para legendas na exportação) |
| `embeddings/` | `base.py` (Protocol `EmbeddingBackend`, `WindowInput`) · `embeddinggemma_client.py` (backend real, lazy, thread-safe) · `video_embedder.py` (monta janelas, batches) |
| `search/` | `vector_store` (LanceDB, uma tabela por dimensão, filtra por modelo) · `query_engine` · `interval_merger` (agrupar < 3 s, expandir < 8 s, dividir > 60 s) · `moment_finder` (limiar efetivo = max(0,65, melhor − `relative_margin` 0,04); score, ranking, completar até 10 resultados marcando os abaixo do limiar) |
| `clipping/` | `vertical_formatter` (corta bordas pretas gravadas, depois smart crop por movimento / centro / blur_pad; `analyze_framing` faz uma passada só para bordas e foco) · `clip_builder` (MP4 H.264+AAC, `-map_metadata -1`, nome `gta6_clip_<data>_<HH-MM-SS>_score093.mp4`) · `thumbnail_generator` · `subtitle_burner` (SRT; ffmpeg roda com cwd na pasta do SRT para evitar escape de `C:`) |
| `publishing/approval_queue.py` | máquina de estados: candidate → approved → exported → published(manual) → archived; rejected |
| `pipeline/` | `indexer` (estágios com progresso persistido no SQLite) · `jobs` (1 worker em thread) |
| `storage/` | SQLite (`database.py` schema, `models.py` dataclasses, `repositories.py`) |
| `services.py` | `ClippingService`: única fachada usada pela UI e pela CLI |
| `ui/` | Streamlit: `streamlit_app.py` + `components/{library,search,queue,about,common}.py`, `i18n.py` (en/pt) |
| `cli.py` | `clipping-machine` com os subcomandos `run`, `doctor`, `download-model`, `index`, `search` |

Desvios conscientes da estrutura da spec (documentar em `docs/ARCHITECTURE.md`): pacote em
`src/clipping_machine/` (instalável); sem `youtube_publisher.py` (regra de não publicar);
`text/image/audio_embedder.py` viraram métodos do cliente; `configs/channels.yaml` não existe
ainda; `prompts/` ainda não criado (só faria sentido com um VLM local, fase 2).

## Fatos verificados sobre o EmbeddingGemma 2 (não "corrija" sem medir de novo)

Versões instaladas: sentence-transformers 6.3.0, transformers 5.23.0, torch 2.15.0+cpu,
lancedb 0.41, streamlit 1.67, scenedetect 0.7.3, numpy 2.5, pandas 3.1.

- Toda entrada passa pelo chat template do modelo (modalidade `message`). O `prompt_name="SearchQuery"` vira mensagem de sistema, inclusive para texto puro. Mídia vai **sem** prompt.
- Vídeo: passar **lista de PIL.Image RGB do mesmo tamanho** (o cliente redimensiona). Caminho de arquivo de vídeo **não funciona** neste ambiente (sem torchcodec/torchvision video). Sempre `processing_kwargs={"video": {"do_sample_frames": False, "add_timestamps": False, "max_soft_tokens": N}}`.
- Áudio: `{"array": ndarray 1-D float32, "sampling_rate": 16000}`. A biblioteca **não reamostra** nem valida a taxa; `prepare_audio` recusa ≠ 16 kHz. Tensor torch falha. Máx. 30 s.
- dtype via `model_kwargs={"dtype": ...}` (não `torch_dtype`). **Nunca float16** (gera NaN). CPU = float32.
- Ordem das chaves do dict define a ordem no prompt: `{"video": ..., "audio": ...}`.
- Similaridade cosseno texto↔vídeo fica numa faixa estreita: ~0,50–0,60 sem relação, ~0,70–0,78 correspondência clara. O limiar da spec (0,65) é aplicado ao **cosseno bruto** e funciona. Para o score, o cosseno é recalibrado com `floor 0.55 / ceiling 0.80` (`search.calibration`).
- Áudio muda pouco o embedding da janela (cos ~0,97 com vs. sem áudio).
- Tags: exigem z-score ≥ 1 dentro do vídeo **e** cosseno ≥ 0,65 (sem o piso absoluto, vídeo sem ação recebia tags aleatórias). Reação: cosseno mapeado de [0,62, 0,76] para [0, 1].
- Velocidade na máquina do mantenedor (Ryzen 7 5825U, sem GPU NVIDIA, 8 threads): ~6 s/janela com 4 frames a 70 tokens; 12 s com 140 tokens. `vision_tokens: auto` = 70 em CPU, 140 em GPU (acurácia top-1 igual no teste). Vídeo de 96 s indexou em 113 s (~1,2x tempo real). Carregar o modelo leva ~16–50 s; depois a busca leva ~0,2 s (o `warm_up` tira o carregamento da métrica de latência).
- Em gameplay real (FF8, 8 min 50 s, 88 janelas) a faixa fica mais cheia: consultas genéricas tinham até 50 janelas acima de 0,65, momentos barulhentos pouco relacionados passavam na frente da cena certa e janelas vizinhas viravam blocos de 40–56 s. Por isso existe `search.relative_margin` (0,04): o limiar efetivo é `max(min_similarity, melhor − 0,04)` e o score só reordena momentos de relevância parecida. Entre esses, o mais barulhento ainda tende a ganhar (fórmula da spec; a CLI tem `--sort similarity`).
- Bordas pretas: a gravação de FF8 tinha a imagem em 1440x~1014 dentro de 1920x1080 (barras com luma ~0–4). `analyze_framing` detectou x 246–1674, y 36–1038, igual ao `cropdetect` do FFmpeg com a margem de 1 px de análise. Jogos atuais em tela cheia não têm borda (a detecção não faz nada), mas cutscenes com letterbox e canvas do OBS têm.
- As tags de `tags.yaml` são de GTA; em outro jogo saem sem sentido (paraquedas, briga em FF8). É esperado: o usuário edita o arquivo.
- Teste com 3 vídeos de GTA 6 (material da Rockstar posto pelo usuário na raiz, só para teste local; **nunca** usar em README/GIF): 15 consultas com gabarito → top-1 10/15, top-3 11/15, 2/2 consultas negativas sem resultado; o `relative_margin` não mudou o top-1 e deixou os intervalos mais justos. Erros = cenas curtas ou escuras.
- Smart crop sem limite cortava o carro/personagem em GTA (o cenário se move, o jogador fica no centro). Agora `limit_focus` mantém o centro da tela na metade central do crop (`export.smart_crop_center_band: 0.5`). Na mira por cima do ombro o personagem fica fora de qualquer crop 9:16: usar `blur_pad`.
- Tags por z-score davam "paraquedas" num escritório escuro (conceito raro no vídeo ganha z alto com pouco). Agora a tag também precisa estar entre os 3 conceitos mais próximos da janela (`tags.max_rank: 3`); "paraquedas" sumiu das 8 janelas. Tags continuam heurísticas ("barco" dentro de um carro).
- Pesos em cache: `~/.cache/huggingface/hub/models--google--embeddinggemma-2` (sem symlinks no Windows; aviso inofensivo).

## Convenções de código

- Tipagem completa; `mypy --strict` deve passar. `ruff` (linha 100; `ui/i18n.py` isento de E501).
- Não nomeie métodos de repositório como `list` (quebra o mypy nas anotações `list[...]`): use `all()`, `for_video()`, `find()`.
- Strings de UI sempre em `ui/i18n.py` (en e pt). Mensagens de exceção em inglês.
- Streamlit 1.67: use `width="stretch"` (não `use_container_width`).
- Novos sinais/limiares vão para `configs/*.yaml` com validação em `config.py`.

## Ambiente de desenvolvimento e cuidados

- Windows 11; ferramentas Bash (Git Bash) e PowerShell 5.1. FFmpeg 8.1 no PATH. `uv` instalado. Docker Desktop instalado. **`gh` não instalado**; há credencial do GitHub no Git Credential Manager.
- **RAM: 40 GB (≈18 GB livres em uso normal), sem GPU NVIDIA.** Não rode mais de um processo que carregue o modelo ao mesmo tempo; o gargalo é CPU.
- Um hook ("Blast Radius") bloqueia comandos com `rm -rf`. Não tente contorná-lo: use pastas novas (ex.: `data_e2e_2`) em vez de apagar.
- Heredocs bash com aspas aninhadas quebram: para edições em lote, escreva um script `.py` no scratchpad e execute-o.
- O usuário fala português (pt-BR): responda em pt-BR.
- Para tarefas simples de leitura e coleta de informações, use subagentes (Haiku ou Opus), a pedido do usuário.

## Estado atual (2026-11-14) e próximos passos

**v0.1.0 publicada**: https://github.com/RodrigoFlorencio86/GTA6-Clipping-Machine (público, branch `main`).

Verificado nesta versão:

- `pytest`: 193 testes passam (Windows 11, Python 3.13, FFmpeg 8.1); `pytest -m model` (modelo real) passa.
- CI no GitHub Actions verde: Ubuntu com Python 3.11, 3.12 e 3.13 e Windows com Python 3.13 (lint, mypy, pytest). No 3.11 o NumPy instalado é o 2.4, cujas anotações devolvem `Any` em `np.clip`/`np.sqrt`: converta com `np.asarray(..., dtype=...)` anotado.
- `ruff check`, `ruff format --check` e `mypy --strict` limpos.
- Teste ponta a ponta com o modelo real (CLI `index` + `search`) num vídeo sintético de 96 s com 8 cenas: as consultas caem nas cenas certas, uma consulta sem correspondência não retorna nada acima de 0,65, busca em ~0,2 s.
- Primeiro teste com gameplay real do usuário (Final Fantasy 8, 8 min 50 s, 1080p, salvo na raiz como `2026-11-14 11-11-11.mp4`, ignorado pelo git): indexou em 692 s (~1,3x), busca ~0,3 s, consultas de cenas marcantes acertaram; levou ao limiar relativo e ao corte de bordas pretas. Clipes exportados depois da correção sem faixas pretas (smart e blur_pad).
- UI coberta por `tests/test_ui.py` (`AppTest`): abas, troca de idioma, recusa de link, exigência de confirmação de direitos e fluxo completo importar → indexar → buscar → gerar clipe → aprovar → exportar.

- Imagem Docker (CPU) compila e sobe; `doctor` roda dentro dela; o app não faz consulta de IP externo (`browser.serverAddress` sempre definido pela CLI; teste em `test_guardrails.py`).

Ainda não verificado pelo mantenedor: GPU CUDA, macOS, faster-whisper com legenda real (o caminho de queima de SRT é testado; a transcrição não), Docker com GPU.

Próximos passos sugeridos (ver também `docs/ROADMAP.md`):

1. Gravar um GIF/screenshot real da UI com gameplay próprio do usuário para o README (aumenta muito a conversão em estrelas).
2. Testar com uma gravação longa real (1–3 h), de preferência de GTA, e ajustar `tags.yaml`, os pesos e `relative_margin` com base no resultado (o 0,04 foi escolhido com um único vídeo real).
3. Criar a release `v0.1.0` no GitHub (tag + notas) quando o usuário validar com gameplay real.
4. Itens da fase 2 do roadmap: ajustar bordas do clipe a cortes de cena e silêncios, descoberta de destaques sem consulta, VLM local para descrições.

Para chamar a API do GitHub sem `gh`: `printf "protocol=https\nhost=github.com\n\n" | git credential fill` devolve o token (escopos `repo`, `workflow`); nunca imprima nem grave o token.
