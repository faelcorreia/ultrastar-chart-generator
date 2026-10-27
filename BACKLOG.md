# Backlog

Objetivo: gerar karaokê UltraStar a partir de qualquer entrada (só MP3, MP3+capa, letra pura, letra sincronizada, KAR/MIDI, MP3+CDG, vídeo), sem importar o caminho. Visão geral da arquitetura: `SongJob` acumula o que se sabe, o planejador escolhe os estágios que faltam.

## Protocolo de sessão

- Sessão com no máximo ~65k de contexto. Ao abrir e dizer "continuar":
  1. Ler este arquivo.
  2. Para cada fatia **Refinado** com dependências concluídas, disparar o subagent `implementador` (Haiku; uma fatia por subagent, uma tacada) até o limite do contexto. A sessão principal (Sonnet) planeja e fecha; o `revisor` (Opus) só entra quando o Rafael pedir revisão, nunca como padrão.
  3. Fatias **A refinar** não vão para subagent: apresentar as perguntas abertas ao Rafael e esperar a decisão dele.
  4. Ao concluir, marcar **Feito** aqui (com o hash do commit) e registrar decisões novas.
- Status: `A refinar` → `Refinado` (só o Rafael move para cá) → `Feito`.
- Cada fatia deve caber em um subagent: escopo fechado, critério de aceite verificável, testes junto.
- Fatias sem commit prévio do estado atual do repositório: ver pré-requisito P0.

## Pré-requisito

| ID | Fatia | Status |
|---|---|---|
| P0 | Commitar o estado atual (`app.py`, `prepare.sh`, `requirements.txt`, `syllables_transcriber.py`, `ctc_word_aligner.py`) | Refinado (ação do Rafael, não vai para subagent) |

Decisão: o Rafael commita. Antes de despachar A1, confirmar que o `git status` está limpo.

## Regras transversais

- **Idiomas:** todo o código precisa funcionar em inglês, português e espanhol. Toda fatia nova traz testes com os três idiomas (aceite obrigatório). Idioma nunca assume `pt` em silêncio: vem da lista, das tags ou da detecção do Whisper; se não der, falha com mensagem clara.

## Épico A: Fundação

| ID | Fatia | Depende | Status |
|---|---|---|---|
| A1 | Modelo de dados: `SongJob`, `Lyrics`, `Line`, `Syllable` (dataclasses, só tipos e testes, sem uso ainda) | P0 | Feito (4cf38e6) |
| A2 | Fazer o pipeline atual do `app.py` passar por `SongJob`, sem mudar comportamento | A1 | Feito (5224e91) |
| A3 | Extrair estágios de `syllables_transcriber.py` (911 linhas) em módulos menores, sem mudar lógica | A2 | Feito (c9fa76e) |
| A4 | `ultrastar_formatter.py` (769 linhas) passa a receber `Lyrics` e ser dividido em módulos, sem mudar saída | A2 | Feito (ad518c7) |
| A5 | Cache por estágio em pasta de trabalho por música, com retomada | A2 | Feito (bb9f92d) |
| A6 | Idioma explícito: resolver idioma por lista/tags/detecção do Whisper, remover o fallback silencioso `pt`, e normalização de texto por idioma (acentos, `ñ`, apóstrofos do inglês) | A1 | Feito (db5a777) |

Decisões (A1): layout em `src/` com `model/ readers/ providers/ stages/ writer/ batch/`; tempo em segundos no modelo, conversão para beats só no writer.

Decisões (A6): idioma vem da lista/tags; se faltar, detecta pelo Whisper e sinaliza no relatório. Comparar Whisper small/medium/large em músicas de en/pt/es e escolher o melhor custo/qualidade (a medição entra nesta fatia como relatório, sem trocar o padrão antes de você ver os números).

Decisão (A5): um JSON por estágio em `output/<música>/.work`; retomada automática.

## Épico B: Entradas de áudio e metadados

| ID | Fatia | Depende | Status |
|---|---|---|---|
| B1 | Aceitar FLAC, WAV, M4A, OGG, OPUS (normalizar via ffmpeg; saída sempre MP3) | A2 | Feito (6105665) |
| B2 | Reader de metadados: tags do arquivo + nome do arquivo, capa embutida | A1 | Feito (f2af66d) |

Decisão (B1): converter tudo para MP3 na saída.

Decisão (B2): nomes no padrão `Artista - Título`; tags do MP3 têm prioridade quando existirem.

## Épico C: Provedores externos

| ID | Fatia | Depende | Status |
|---|---|---|---|
| C1 | Cliente LRCLIB (`/api/get`, `/api/search`), cache em disco, validação por duração | A1 | Feito (f2af66d) |
| C2 | MusicBrainz + Cover Art Archive: metadados e capa | A1 | Feito (f2af66d) |

Decisões (C1): aceitar LRC com duração dentro de ±3 s; preferir sincronizada, usar letra pura só se não houver; API online com cache em disco (sem dump local).

Decisão (C2): MusicBrainz/Cover Art Archive só para o que faltar (ano, álbum, capa); tags e lista têm prioridade.

## Épico D: Letra e alinhamento

| ID | Fatia | Depende | Status |
|---|---|---|---|
| D1 | Parser de LRC para `Lyrics` com tempo por linha | A1 | Feito (3c39a4e) |
| D2 | Alinhar palavras/sílabas dentro da janela de cada linha (CTC restrito) | A3, D1 | Feito (cb8838a) |
| D3 | Letra pura + áudio: alinhamento forçado sem Whisper | A3 | Feito (cb8838a) |
| D4 | Reconciliar letra baixada com a transcrição do Whisper: alinhamento de sequências, detectar refrão faltando, verso deslocado e trecho perdido; corrigir ordem/repetições e calcular confiança por música | D2, D3 | Feito (2a2e2e9) |
| D5 | Validar a sincronização de letras de fontes externas (LRC, KAR, CDG) contra o áudio, com duas medições independentes (Whisper e CTC) e correção: offset global, deriva ou linha a linha | D2, D4 | Feito (e1ca24d) |

Decisões (D1-D3): D1 corrige offset global do LRC estimando pelas palavras do Whisper e só aplica se melhorar o casamento; CTC (MMS, en/pt/es) alinha palavras dentro das janelas de linha e o MFA segue nas sílabas.

Ideia (D5): o áudio é a verdade. Para cada linha da fonte externa, medir o tempo real com dois estimadores independentes (palavras do Whisper e alinhamento forçado CTC no vocal separado). Se os dois concordam entre si e discordam da fonte, a fonte está errada e se corrige (offset global, deriva linear ou realinhar a linha). Se os dois discordam entre si, a medição é incerta: não decide sozinho, marca "incerto" e a música vai para "revisar". Se concordam com a fonte, mantém. Saída: relatório por música com porcentagem de linhas OK, deslocadas e incertas.

Decisões (D5): início de linha com até 0,5 s de diferença é OK; Whisper e CTC discordando entre si marca "incerto" e vai para revisar (relatório com os três tempos); correção automática do mais simples ao mais fino (offset global, depois deriva linear, depois realinhar só as linhas ruins), escolhendo a que mais melhora o casamento. G2: render MIDI só como último recurso, MP3 original tem prioridade.

Decisões (D4): refrão faltando é copiado do refrão anterior e confirmado pelo CTC na janela do áudio (se não bater, sinaliza); abaixo de 80% de palavras casadas a música vai para "revisar". Idiomas: inglês, português e espanhol.

Perguntas abertas:

## Épico E: Planejador

| ID | Fatia | Depende | Status |
|---|---|---|---|
| E1 | `planner.py`: dado um `SongJob`, lista os estágios necessários e executa com cache | A5, D2, D3, C1 | Feito (1e0dbab) |

Decisão (E1): só letra e sem áudio vai para a fila "aguardando áudio": guarda a letra, registra no relatório e retoma quando o áudio aparecer.

## Épico F: Lote

| ID | Fatia | Depende | Status |
|---|---|---|---|
| F1 | CLI de lote: lista `artista;título;ano`, localiza o áudio, processa em fila | E1 | Feito (24535d9) |
| F2 | Relatório do lote (entrou, falhou, por quê) e retomada | F1 | Feito (c3a1b92) |

Decisão (F2): relatório e estado de retomada em JSON (entrou, falhou, revisar, aguardando áudio, com motivo por música).

Decisões (F1): o áudio é localizado numa pasta indicada, casando `Artista - Título` com tolerância a acento e maiúsculas; músicas que já existem em `/run/media/ghost/ROMS/ultrastar-songs` (artista+título normalizados) são puladas por padrão, com flag para forçar; uma música por vez.

## Épico G: Formatos de karaokê pronto

| ID | Fatia | Depende | Status |
|---|---|---|---|
| G1 | Reader KAR/MIDI: letra, tempo por sílaba e melodia | A1 | Feito (3c39a4e) |
| G2 | Render de áudio MIDI (FluidSynth + soundfont) | G1 | Feito (a1100ac) |
| G3 | Parser CDG (+ zip MP3+CDG): extrair letra e tempo | A1 | Feito (3c39a4e) |
| G4 | Vídeo instrumental com letra via OCR | A1 | Feito (a1100ac) |

Decisões (G): G1 usa a melodia do MIDI quando existir, senão pitch tracking; G3 usa o parser dos comandos de realce do CDG para o tempo e OCR só para ler o texto; G4 fica por último.

## Épico H: Fontes e download

Fora do código por enquanto. Discussão com o Rafael sobre onde obter KAR/CDG/áudio e o que é legítimo automatizar. A sessão Produtividade entrega a lista (formato a definir). Sem fatias até a decisão.

## Épico I: Integração (proposta, A refinar)

Todas as fatias acima foram testadas só com fakes. Falta ligar tudo de ponta a ponta.

| ID | Fatia | Depende | Status |
|---|---|---|---|
| I1a | Ligar ao planner os estágios reais vocals, pitches e beats (hoje atrás da API de dicionários do `legacy.py`) | E1 | Feito (cb303be) |
| I1b | Ligar ao planner os estágios reais syllables e write | I1a | Feito (eec4751) |
| I2 | Ligar B1 (`normalize_audio`), D4 (`reconcile`) e D5 (`sync_validate`) ao plano, com confirmador CTC e estimadores reais | I1 | Feito (86f0c3b) |
| I3 | Caminho Whisper quando o LRCLIB não acha letra (hoje o job falha). Aceite: LRCLIB sem resultado → Whisper transcreve → segmenta com I5 → grava com status "revisar" e motivo no relatório; testes com fakes em en/pt/es | I1a, I1b, I5 | Feito (4fa2a68) |
| I4 | Rodar de verdade com áudio real em en/pt/es e registrar os resultados | I1 | Feito (89e110e) — ver `docs/i4-results.md` |
| I5 | Segmentação em versos e estrofes (pausas do vocal, pontuação, refrão, máx. 12 sílabas) | I1 | Feito (b0b417e) |
| I6 | Matcher aproximado (fuzzy) no D5 no lugar do prefixo exato de 3 palavras, para letra mal ouvida pelo Whisper. Medir antes/depois com `scripts/i4_compare.py` | I4 Aceite: métrica melhor que antes no `scripts/i4_compare.py` sobre o lote do sdcard, sem regressão nas demais músicas; testes en/pt/es. | Feito (64423bc) |
| I7 | Busca de título no LRCLIB tolerante a prefixos/variações (ex.: "12 - Tribalistas - Já Sei Namorar"): limpar o nome (prefixo numérico, tags, variações em cascata). Testes en/pt/es | I4 | Feito (08e2993) |
| I8 | Whisper medium como único modelo do caminho Whisper (sem large-v3). Quando não há letra, avisar claramente na UI/CLI/relatório que fornecer a letra é fortemente recomendado (qualidade muito melhor) e marcar "revisar" se a cobertura de linhas for baixa. Medir small vs medium com `scripts/compare_whisper_models.py` no português denso e no Jota Quest. Testes en/pt/es | I4 | Feito (7239432) |
| I9 | Investigar Green Day - She (5% das linhas OK): referência diferente ou CTC travando em repetições | I4 Aceite: métrica melhor que antes no `scripts/i4_compare.py` sobre o lote do sdcard, sem regressão nas demais músicas; testes en/pt/es. | Feito (fd9c89f) |
| I10 | D4 insere refrão copiado com tempo errado (visto na Avril) | I4 Aceite: métrica melhor que antes no `scripts/i4_compare.py` sobre o lote do sdcard, sem regressão nas demais músicas; testes en/pt/es. | Feito (8489b30) |
| I11 | Capitalização mínima: maiúscula só no início de cada verso, resto minúsculo (exceto "I" em inglês); não corrigir ortografia/gramática (grafia artística é válida). Testes en/pt/es | I4 | Feito (1d901d2) |
| I12 | Til (`~`) para nota longa/segurada: quando a voz segura a vogal, a nota continua com `~` em vez de sílaba nova. Testes en/pt/es | I4 | Feito (597b52f) |
| I13 | Dessincronização "aleatória" e pausa antes da palavra ser cantada (visto em Pegasus Fantasy, es): geral está ok, mas linhas soltam do áudio. Diagnosticar com `i4_compare.py` por linha | I4 Aceite: métrica melhor que antes no `scripts/i4_compare.py` sobre o lote do sdcard, sem regressão nas demais músicas; testes en/pt/es. | Feito (d1bd158) |
| I14 | Vocal separado do Jota Quest - O vento saiu mudo (média -84 dB, 0 palavras): investigar a falha na separação de vocal e detectar stem mudo com fallback | I4 | A refinar |
| I15 | Reavaliar Whisper medium vs small: medium não melhorou o WER no português cantado (Tribalistas 0,973 vs 0,973; Jota Quest 0,929 small vs 0,980 medium) e é 6-7x mais lento | I8 | A refinar |

Decisão (I9): a referência do Green Day está certa (mesmo áudio, correlação com lag 0); o LRCLIB tem deslocamento constante de ~2,4 s e o CTC global erra no começo (o vocal separado fica mudo de 0 a 29 s). Com o Whisper surdo, `sync_validate` ajusta um modelo robusto (RANSAC de offset/deriva, tolerância 1,25 s, ≥30% de acordo) e só confia no CTC da linha se estiver a até 1,5 s do modelo. Green Day 2,7% → 18,9% (precisão 5% → 35%); Avril igual (52%). O restante é o limite do CTC em canto, fora de 0,5 s.

Decisão (I7): `src/providers/name_clean.py` gera variantes (artista, título) em cascata do mais específico ao mais frouxo: nome original; sem prefixo numérico (`12 - `, `12. `, `12 `) e sem tags `[..]`/parênteses de ruído (Live, Official, Ao Vivo, Remastered...); sem `feat.`; sem qualquer parêntese. `LrclibClient.find` tenta cada uma e para na primeira com letra (o nome exato sempre vai primeiro, então títulos como "99 Luftballons" seguem funcionando). `parse_filename` também aceita os prefixos `01. `, `01 ` e `01-`. Parênteses sem palavra de ruído (ex.: "(Interlude)") só saem na última variante.

Decisão (I10): o D4 só insere um refrão copiado quando o CTC confirma a janela do áudio; se não confirmar (ou não houver CTC) ele é sinalizado e não é inserido. Novo guarda: janela de gap maior que 2 s por palavra do trecho é rejeitada, porque o confirmador aceitava qualquer janela com durações plausíveis (Avril: 4 linhas "Yeah, yeah, yeah" copiadas para uma janela de 62 s, esticadas sobre outro material). Medido na Avril com a transcrição do Whisper que dispara o caso: linhas OK 60% → 72%, precisão 68% → 82%. Green Day (18,9%) e Manu Chao (37,5%) não têm refrão faltando e ficam iguais.

Decisão (I11): `minimal_case` (`src/core/text_norm.py`) aplicada só ao caminho Whisper (`whisper_lyrics`); letra pura/LRC mantém a caixa original para não desfigurar nomes próprios.

Decisões (I): ordem fica a critério do Claude (sugestão: I1, I4 cedo, depois I2, I3). I3: sem letra no LRCLIB, cai no Whisper sozinho e marca a música como "revisar".

Problema levantado pelo Rafael (I5, Refinado): separar o texto em versos/linhas. Letra pura ou transcrição do Whisper vira um "linguição" sem quebras; não vale entregar assim. Precisa de uma estratégia de segmentação em linhas/estrofes (pausas do áudio, pontuação, repetição de refrão, comprimento de linha) e de um critério de qualidade verificável. Perguntas abertas: usar pausas do vocal como fonte principal? Limite de sílabas por linha? Testar em letras reais en/pt/es antes de decidir.


Decisões (I2, 2026-10-26): estimadores reais = Whisper + CTC/MMS no vocal separado; `normalize_audio` só quando a entrada não é MP3; ordem reconcile (D4) → sync_validate (D5). I4: Rafael liberou escolher um lote de áudios no sdcard e iterar variações sozinho; ele ouve quando o Claude pedir.

Feedback do Rafael ao ouvir a I4 no UltraStar Play (2026-10-26): Pegasus Fantasy ok mas pouco alinhada, com pausa em palavra ainda não cantada e dessincronia ocasional; falta controle de maiúsculas/minúsculas (sem corrigir ortografia, pois grafia artística existe); quer o `~` de volta para nota longa. Pastas de teste: `/run/media/ghost/Dados/ultrastar-i4-ouvir` (já em SongDirs do UltraStar Play; backup `Settings.json.bak-i4`). Regra de disco: modelos em `/run/media/ghost/Dados/LLM/`.

Decisões (poll, 2026-10-26): I1 dividida em I1a/I1b. I4: Rafael indica os arquivos de áudio (segue A refinar até lá). I5 (Refinado): fonte principal = pausas do vocal, com pontuação e repetição de refrão como apoio; limite de 12 sílabas por linha, configurável; critério verificável: nenhuma linha acima do limite e quebra em pausas longas; testes em en/pt/es.

Decisões (refino, 2026-10-26): todas as fatias I6-I10, I13 refinadas pelo Rafael. Aceite de qualidade = métrica no `i4_compare`. Ordem: I13, I6, I9, I10, I8, I7. Máquina: CPU 8 núcleos, 39 GB RAM, Radeon sem CUDA (Whisper só em CPU).

Decisão (I8, 2026-10-26): só Whisper medium (CPU int8); letra de entrada fortemente recomendada, com aviso.

Decisão (I8): `WHISPER_MODEL = "medium"` (`src/core/whisper_model.py`) é o único modelo (transcrição, detecção de idioma, checagem de sílabas); `large-v3` removido do script/docs. Sem letra, o caminho Whisper agora avisa "providing the lyrics is STRONGLY RECOMMENDED" (motivo do relatório "revisar", nota do job, stderr da CLI) e acrescenta "low line coverage" ao motivo de revisão quando as linhas cobrem <25% da duração ou são <4. Medição (`compare_whisper_models.py`, mix completo, pt, WER vs LRCLIB, CPU int8): Tribalistas small 0,973 / medium 0,973 (37 s / 245 s); Jota Quest small 0,929 / medium 0,980 (43 s / 295 s). Ambos falham em canto (WER ~1, laços de repetição) e medium é 6-7x mais lento sem ganho aqui; o stem vocal do Jota Quest está mudo (-84 dB). Mantida a decisão do Rafael (medium); reavaliar se o custo pesar.
