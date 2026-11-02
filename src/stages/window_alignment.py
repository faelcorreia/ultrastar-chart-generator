"""D2/K2b: alinhamento de sílabas dentro da janela de cada linha, pelo almighty-listener.

Adaptador: monta `lines` a partir da letra sincronizada (LRC), com a janela de cada linha
(início até o início da próxima), chama `listen(<áudio completo>, want=[transcript, alignment])`
e converte o resultado para `Lyrics`. Quem alinha é o almighty: ele separa a voz (stems, com
cache compartilhado), alinha cada linha só na sua janela com tempos absolutos e, com `alt_audio`
(I13), tenta também no mix e fica com o início mais perto da janela.

Leitura do resultado:
- transcript.json (versão 2): `line` por palavra e `failed_lines`. É dele que sai o agrupamento
  por linha e a lista de falhas, em índices da letra.
- alignment.json: as sílabas de cada palavra, na mesma ordem das palavras do transcript.

`line_windows` continua aqui (o ultrastar define as janelas). Linha sem palavras (só pontuação)
não vai para o almighty e sai como estava, sem sílabas.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.model.lyrics import Line, Lyrics, Syllable
from src.stages.almighty_runners import AlmightyError, ListenFn, _fetch, _load_json
from src.stages.word_alignment import normalize_language, split_words

WANT = ("transcript", "alignment")
TRANSCRIPT_LINES_VERSION = 2


@dataclass
class WindowAlignment:
    lyrics: Lyrics
    failed_lines: List[int] = field(default_factory=list)


def line_windows(lyrics: Lyrics, audio_duration: Optional[float] = None,
                 tail: float = 8.0) -> List[Optional[tuple]]:
    """(start, end) per line; None for lines without a start time."""
    windows: List[Optional[tuple]] = []
    lines = lyrics.lines
    for i, line in enumerate(lines):
        if line.start is None:
            windows.append(None)
            continue
        nxt = next((l.start for l in lines[i + 1:] if l.start is not None),
                   None)
        if nxt is not None:
            end = nxt
        elif audio_duration is not None:
            end = audio_duration
        else:
            end = line.end if line.end and line.end > line.start \
                else line.start + tail
        if audio_duration is not None:
            end = min(end, audio_duration)
        windows.append((line.start, max(end, line.start + 0.05)))
    return windows


def _contract(message: str, name: str) -> AlmightyError:
    return AlmightyError(f"resultado do almighty fora do contrato em {name!r}: {message}")


def _syllables(word: Dict[str, Any], last_in_line: bool, name: str) -> List[Syllable]:
    """Sílabas de uma palavra do alignment, com o espaço final de cada uma (como o ultrastar)."""
    parts = word.get("syllables") or []
    if not parts:
        raise _contract(f"palavra {word.get('text')!r} sem sílabas", name)
    out: List[Syllable] = []
    for index, part in enumerate(parts):
        text = str(part["text"])
        if index == len(parts) - 1 and not last_in_line:
            text += " "
        out.append(Syllable(text=text, start=float(part["start"]), end=float(part["end"])))
    return out


def _to_lyrics(lyrics: Lyrics, sent: List[Tuple[int, List[str]]], lang: str,
               transcript: Dict[str, Any], alignment: Dict[str, Any], name: str) -> WindowAlignment:
    """Monta o Lyrics: linhas enviadas com sílabas, as demais como estavam."""
    if transcript.get("schema_version") != TRANSCRIPT_LINES_VERSION:
        raise _contract(
            f"transcript sem linhas (schema_version {transcript.get('schema_version')!r}, "
            f"esperado {TRANSCRIPT_LINES_VERSION})", name)
    t_words = transcript.get("words")
    a_words = alignment.get("words")
    if not isinstance(t_words, list) or not isinstance(a_words, list):
        raise _contract("words ausente", name)
    expected = sum(len(words) for _, words in sent)
    if len(t_words) != expected or len(a_words) != expected:
        raise _contract(
            f"{len(t_words)} palavra(s) no transcript e {len(a_words)} no alignment, "
            f"esperado {expected}", name)
    failed_sent = set(transcript.get("failed_lines") or [])

    built: Dict[int, Line] = {}
    failed: List[int] = []
    pos = 0
    for k, (index, words) in enumerate(sent):
        chunk_t = t_words[pos:pos + len(words)]
        chunk_a = a_words[pos:pos + len(words)]
        pos += len(words)
        syllables: List[Syllable] = []
        for j, (tw, aw) in enumerate(zip(chunk_t, chunk_a)):
            if tw.get("line") != k or tw.get("text") != words[j] or aw.get("text") != words[j]:
                raise _contract(f"palavra {words[j]!r} fora de ordem na linha {index}", name)
            syllables.extend(_syllables(aw, j == len(words) - 1, name))
        built[index] = Line(text=" ".join(words), syllables=syllables)
        if k in failed_sent:
            failed.append(index)

    out_lines: List[Line] = []
    for index, line in enumerate(lyrics.lines):
        if index in built:
            out_lines.append(built[index])
        else:
            out_lines.append(Line(text=line.text, start=line.start, end=line.end))
    return WindowAlignment(Lyrics(lines=out_lines, language=lang, source=lyrics.source),
                           failed)


def align_in_windows(
    lyrics: Lyrics,
    audio_path: Path,
    language: Optional[str] = None,
    *,
    work: Path,
    listen: Optional[ListenFn] = None,
    audio_duration: Optional[float] = None,
    alt_audio: bool = True,
) -> WindowAlignment:
    """Return a new Lyrics with word/syllable times, each line aligned in its own window.

    ``audio_path`` is the full track: the almighty separates the voice itself. ``work`` is the
    song's ``.work`` folder (the almighty cache sits next to the songs). A line whose alignment
    fails keeps even timing inside its window and its index is reported in ``failed_lines``.
    With ``alt_audio`` (I13) each line is also aligned on the mix and the closer start wins.
    """
    lang = normalize_language(language or lyrics.language)
    if not lyrics.is_synced:
        raise ValueError("align_in_windows needs line times (synced lyrics)")

    sent: List[Tuple[int, List[str]]] = []
    lines_opt: List[Dict[str, Any]] = []
    for index, (line, window) in enumerate(
            zip(lyrics.lines, line_windows(lyrics, audio_duration))):
        words = split_words(line.text)
        if not words:
            continue
        sent.append((index, words))
        lines_opt.append({
            "text": " ".join(words),
            "window": None if window is None else [float(window[0]), float(window[1])],
        })
    if not sent:
        return WindowAlignment(
            Lyrics(lines=[Line(text=l.text, start=l.start, end=l.end) for l in lyrics.lines],
                   language=lang, source=lyrics.source),
            [])

    work = Path(work)
    name = work.parent.name
    options = {"transcript": {"lines": lines_opt, "language": lang,
                              "alt_audio": bool(alt_audio)}}
    paths = _fetch(listen, Path(audio_path), WANT, WANT, name, work, "a letra",
                   options=options)
    transcript = _load_json(paths["transcript"], "transcript", name)
    alignment = _load_json(paths["alignment"], "alignment", name)
    return _to_lyrics(lyrics, sent, lang, transcript, alignment, name)
