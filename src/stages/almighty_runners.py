"""K1: runners do almighty-listener para vocals, pitches e beats.

Adaptador: cada runner tem a mesma assinatura de `audio_analysis.py`,
``runner(song, name, work) -> dict``, chama ``listen()`` do pacote almighty-listener,
lê o artefato e regrava no ``.work`` com os nomes antigos. writer e formatter não mudam.

- vocals: ``stems`` sobre a faixa completa -> ``<nome> - vocals.wav`` e ``- instrumental.wav``
- pitches: ``pitch`` sobre a faixa completa (o almighty deriva a voz dos stems; os stems saem do
  cache do estágio vocals) -> ``<nome> - pitches.json`` no formato antigo (``times``,
  ``frequencies``, ``midi_notes``, ``voiced_prob``; quadro sem voz = 0.0, não null)
- beats: ``beats`` sobre o instrumental -> ``<nome> - beats.json`` (``bpm``, ``beats``, e
  ``downbeats``, ``engine``, ``meter`` que o writer ignora)

O cache do almighty fica em ``<saída>/.almighty-cache`` (``work`` é ``<saída>/<música>/.work``),
compartilhado entre músicas. O ``listen`` é injetável (``listen=``) para os testes; sem ele, o
pacote é importado só quando o runner roda, para não pesar a importação do planejador.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

INSTALL_HINT = "instale as dependências do requirements.txt (almighty-listener @ v0.1.0)"
CACHE_DIR_NAME = ".almighty-cache"

ListenFn = Callable[..., Any]


class AlmightyError(RuntimeError):
    """almighty-listener ausente, falha ao analisar o áudio ou artefato fora do contrato."""


def _default_listen() -> ListenFn:
    try:
        from almighty_listener.api import listen
    except ImportError as exc:
        raise AlmightyError(f"almighty-listener não instalado ({exc}); {INSTALL_HINT}") from exc
    return listen


def cache_dir_for(work: Path) -> Path:
    """Cache compartilhado do almighty: uma pasta acima das músicas da saída."""
    return Path(work).parent.parent / CACHE_DIR_NAME


def _fetch(listen: Optional[ListenFn], audio: Path, want: Sequence[str],
           artifacts: Sequence[str], name: str, work: Path, what: str,
           options: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """Roda o almighty sobre `audio` e devolve o caminho de cada artefato pedido.

    `options` (mapa estágio -> opções) vai para o `listen`; K2b usa para `transcript`.
    `listen` é lazy no pacote: a execução e os erros de estágio acontecem no `get`, por isso
    as duas chamadas ficam dentro do mesmo try.
    """
    fn = listen if listen is not None else _default_listen()
    try:
        result = fn(
            audio,
            want=list(want),
            options=options,
            cache_dir=cache_dir_for(work),
            work_dir=Path(work) / ".almighty",
        )
        return {artifact: Path(result.get(artifact)) for artifact in artifacts}
    except Exception as exc:
        raise AlmightyError(
            f"almighty-listener falhou ao analisar {what} de {name!r}: {exc}"
        ) from exc


def _copy(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return dest


def _load_json(path: Path, artifact: str, name: str) -> Dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AlmightyError(f"{artifact} de {name!r} ilegível: {exc}") from exc
    if not isinstance(data, dict):
        raise AlmightyError(f"{artifact} de {name!r} fora do contrato: raiz não é objeto")
    return data


def _midi(hz: float) -> float:
    """Mesma conta do librosa.hz_to_midi (A4 = 440 Hz = MIDI 69), com 2 casas."""
    return round(69.0 + 12.0 * math.log2(hz / 440.0), 2)


def legacy_pitches(data: Dict[str, Any], name: str = "") -> Dict[str, Any]:
    """Converte o pitch.json do almighty no formato que o writer lê.

    Quadro sem voz (`f0_hz` null) vira 0.0 em `frequencies` e `midi_notes`; `voiced_prob`
    recebe a confiança do quadro (ou 0.0 se faltar).
    """
    try:
        frames = data["frames"]
        times, frequencies, midi_notes, voiced_prob = [], [], [], []
        for frame in frames:
            hz = frame.get("f0_hz")
            hz = float(hz) if hz is not None else 0.0
            if not math.isfinite(hz) or hz < 0:
                raise AlmightyError(f"pitch de {name!r} com f0_hz inválido: {hz}")
            confidence = frame.get("confidence")
            times.append(float(frame["time"]))
            frequencies.append(hz)
            midi_notes.append(_midi(hz) if hz > 0.0 else 0.0)
            voiced_prob.append(float(confidence) if confidence is not None else 0.0)
    except (KeyError, TypeError, ValueError) as exc:
        raise AlmightyError(f"pitch de {name!r} fora do contrato: {exc}") from exc
    return {
        "times": times,
        "frequencies": frequencies,
        "midi_notes": midi_notes,
        "voiced_prob": voiced_prob,
    }


def almighty_separate(song: Dict[str, Any], name: str, work: Path,
                      listen: Optional[ListenFn] = None) -> Dict[str, Any]:
    """Stems vocals/instrumental em WAV, copiados para o `.work` com os nomes antigos."""
    work = Path(work)
    paths = _fetch(listen, Path(song["full_track"]), ["stems"],
                   ("stems.vocals", "stems.instrumental"), name, work, "os stems")
    return {
        "vocals": _copy(paths["stems.vocals"], work / f"{name} - vocals.wav"),
        "instrumental": _copy(paths["stems.instrumental"],
                              work / f"{name} - instrumental.wav"),
    }


def almighty_pitches(song: Dict[str, Any], name: str, work: Path,
                     listen: Optional[ListenFn] = None) -> Dict[str, Any]:
    """Contorno de f0 no formato antigo de pitches.json."""
    work = Path(work)
    paths = _fetch(listen, Path(song["full_track"]), ["pitch"], ("pitch",), name, work,
                   "o pitch")
    data = _load_json(paths["pitch"], "pitch", name)
    out = work / f"{name} - pitches.json"
    out.write_text(json.dumps(legacy_pitches(data, name), ensure_ascii=False),
                   encoding="utf-8")
    return {"pitches": out}


def almighty_beats(song: Dict[str, Any], name: str, work: Path,
                   listen: Optional[ListenFn] = None) -> Dict[str, Any]:
    """Tempo e beats do instrumental (o beats do almighty lê a entrada, não os stems)."""
    work = Path(work)
    audio = Path(song.get("instrumental") or song["full_track"])
    paths = _fetch(listen, audio, ["beats"], ("beats",), name, work, "os beats")
    data = _load_json(paths["beats"], "beats", name)
    try:
        payload = {
            "bpm": float(data["bpm"]),
            "beats": [float(t) for t in data["beats"]],
            "downbeats": [float(t) for t in data.get("downbeats", [])],
            "engine": data.get("engine"),
            "meter": data.get("meter"),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise AlmightyError(f"beats de {name!r} fora do contrato: {exc}") from exc
    out = work / f"{name} - beats.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"beats": out}
