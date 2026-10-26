# Backlog

Objetivo: gerar karaokê UltraStar a partir de qualquer entrada (só MP3, MP3+capa, letra pura, letra sincronizada, KAR/MIDI, MP3+CDG, vídeo), sem importar o caminho. Visão geral da arquitetura: `SongJob` acumula o que se sabe, o planejador escolhe os estágios que faltam.

## Protocolo de sessão

- Sessão com no máximo ~65k de contexto. Ao abrir e dizer "continuar":
  1. Ler este arquivo.
  2. Para cada fatia **Refinado** com dependências concluídas, disparar um subagent (uma fatia por subagent, uma tacada) até o limite do contexto.
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
| A5 | Cache por estágio em pasta de trabalho por música, com retomada | A2 | Refinado |
| A6 | Idioma explícito: resolver idioma por lista/tags/detecção do Whisper, remover o fallback silencioso `pt`, e normalização de texto por idioma (acentos, `ñ`, apóstrofos do inglês) | A1 | Refinado |

Decisões (A1): layout em `src/` com `model/ readers/ providers/ stages/ writer/ batch/`; tempo em segundos no modelo, conversão para beats só no writer.

Decisões (A6): idioma vem da lista/tags; se faltar, detecta pelo Whisper e sinaliza no relatório. Comparar Whisper small/medium/large em músicas de en/pt/es e escolher o melhor custo/qualidade (a medição entra nesta fatia como relatório, sem trocar o padrão antes de você ver os números).

Decisão (A5): um JSON por estágio em `output/<música>/.work`; retomada automática.

## Épico B: Entradas de áudio e metadados

| ID | Fatia | Depende | Status |
|---|---|---|---|
| B1 | Aceitar FLAC, WAV, M4A, OGG, OPUS (normalizar via ffmpeg; saída sempre MP3) | A2 | Refinado |
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
| D4 | Reconciliar letra baixada com a transcrição do Whisper: alinhamento de sequências, detectar refrão faltando, verso deslocado e trecho perdido; corrigir ordem/repetições e calcular confiança por música | D2, D3 | Refinado |
| D5 | Validar a sincronização de letras de fontes externas (LRC, KAR, CDG) contra o áudio, com duas medições independentes (Whisper e CTC) e correção: offset global, deriva ou linha a linha | D2, D4 | Refinado |

Decisões (D1-D3): D1 corrige offset global do LRC estimando pelas palavras do Whisper e só aplica se melhorar o casamento; CTC (MMS, en/pt/es) alinha palavras dentro das janelas de linha e o MFA segue nas sílabas.

Ideia (D5): o áudio é a verdade. Para cada linha da fonte externa, medir o tempo real com dois estimadores independentes (palavras do Whisper e alinhamento forçado CTC no vocal separado). Se os dois concordam entre si e discordam da fonte, a fonte está errada e se corrige (offset global, deriva linear ou realinhar a linha). Se os dois discordam entre si, a medição é incerta: não decide sozinho, marca "incerto" e a música vai para "revisar". Se concordam com a fonte, mantém. Saída: relatório por música com porcentagem de linhas OK, deslocadas e incertas.

Decisões (D5): início de linha com até 0,5 s de diferença é OK; Whisper e CTC discordando entre si marca "incerto" e vai para revisar (relatório com os três tempos); correção automática do mais simples ao mais fino (offset global, depois deriva linear, depois realinhar só as linhas ruins), escolhendo a que mais melhora o casamento. G2: render MIDI só como último recurso, MP3 original tem prioridade.

Decisões (D4): refrão faltando é copiado do refrão anterior e confirmado pelo CTC na janela do áudio (se não bater, sinaliza); abaixo de 80% de palavras casadas a música vai para "revisar". Idiomas: inglês, português e espanhol.

Perguntas abertas:

## Épico E: Planejador

| ID | Fatia | Depende | Status |
|---|---|---|---|
| E1 | `planner.py`: dado um `SongJob`, lista os estágios necessários e executa com cache | A5, D2, D3, C1 | Refinado |

Decisão (E1): só letra e sem áudio vai para a fila "aguardando áudio": guarda a letra, registra no relatório e retoma quando o áudio aparecer.

## Épico F: Lote

| ID | Fatia | Depende | Status |
|---|---|---|---|
| F1 | CLI de lote: lista `artista;título;ano`, localiza o áudio, processa em fila | E1 | Refinado |
| F2 | Relatório do lote (entrou, falhou, por quê) e retomada | F1 | Refinado |

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
