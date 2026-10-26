"""KAR / MIDI reader (G1): lyrics, per-syllable timing and melody.

Own SMF parser (formats 0 and 1, tempo map, running status). No dependency.
"""
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.model.lyrics import Line, Lyrics, Syllable

_LANG = {"ENG": "en", "ENGL": "en", "EN": "en", "POR": "pt", "PORT": "pt",
         "PT": "pt", "SPA": "es", "SPAN": "es", "ESP": "es", "ES": "es"}


@dataclass
class Note:
    start: float
    end: float
    pitch: int


@dataclass
class KarResult:
    lyrics: Lyrics
    melody: List[Note] = field(default_factory=list)
    title: Optional[str] = None
    artist: Optional[str] = None
    info: List[str] = field(default_factory=list)


@dataclass
class _Track:
    name: str = ""
    events: List[Tuple[int, int, bytes]] = field(default_factory=list)
    # (tick, meta type, data) for meta events
    notes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    # (start tick, end tick, pitch, channel)


def _varlen(data: bytes, pos: int) -> Tuple[int, int]:
    value = 0
    while True:
        b = data[pos]
        pos += 1
        value = (value << 7) | (b & 0x7F)
        if not b & 0x80:
            return value, pos


def _decode(raw: bytes) -> str:
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "replace")


def _parse_track(data: bytes) -> _Track:
    tr = _Track()
    pos, tick, running = 0, 0, 0
    active: Dict[Tuple[int, int], int] = {}
    n = len(data)
    while pos < n:
        delta, pos = _varlen(data, pos)
        tick += delta
        status = data[pos]
        if status < 0x80:
            status = running
        else:
            pos += 1
            if status < 0xF0:
                running = status
        if status == 0xFF:
            mtype = data[pos]
            length, pos = _varlen(data, pos + 1)
            payload = data[pos:pos + length]
            pos += length
            if mtype == 0x03 and not tr.name:
                tr.name = _decode(payload)
            if mtype == 0x2F:
                break
            tr.events.append((tick, mtype, payload))
        elif status in (0xF0, 0xF7):
            length, pos = _varlen(data, pos)
            pos += length
        else:
            kind, ch = status & 0xF0, status & 0x0F
            size = 1 if kind in (0xC0, 0xD0) else 2
            args = data[pos:pos + size]
            pos += size
            if kind == 0x90 and args[1] > 0:
                active.setdefault((ch, args[0]), tick)
            elif kind == 0x80 or (kind == 0x90 and args[1] == 0):
                st = active.pop((ch, args[0]), None)
                if st is not None:
                    tr.notes.append((st, tick, args[0], ch))
    for (ch, pitch), st in active.items():
        tr.notes.append((st, tick, pitch, ch))
    return tr


def _parse_smf(data: bytes) -> Tuple[int, List[_Track]]:
    if data[:4] == b"RIFF":  # RMID wrapper
        i = data.find(b"MThd")
        if i < 0:
            raise ValueError("Not a MIDI file")
        data = data[i:]
    if data[:4] != b"MThd":
        raise ValueError("Not a MIDI file")
    hlen = struct.unpack(">I", data[4:8])[0]
    _fmt, ntracks, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise ValueError("SMPTE time division is not supported")
    pos = 8 + hlen
    tracks: List[_Track] = []
    while pos + 8 <= len(data) and len(tracks) < ntracks:
        cid, size = data[pos:pos + 4], struct.unpack(">I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        pos += 8 + size
        if cid == b"MTrk":
            tracks.append(_parse_track(body))
    return division, tracks


class _TempoMap:
    def __init__(self, division: int, tracks: List[_Track]):
        changes = sorted((t, int.from_bytes(d, "big"))
                         for tr in tracks for t, m, d in tr.events
                         if m == 0x51 and len(d) == 3)
        self.div = division
        self.segs: List[Tuple[int, float, int]] = []  # tick, sec, tempo
        tick, sec, tempo = 0, 0.0, 500000
        self.segs.append((0, 0.0, tempo))
        for t, new in changes:
            sec += (t - tick) * tempo / 1e6 / division
            tick, tempo = t, new
            self.segs.append((tick, sec, tempo))

    def seconds(self, tick: int) -> float:
        seg = self.segs[0]
        for s in self.segs:
            if s[0] <= tick:
                seg = s
            else:
                break
        return seg[1] + (tick - seg[0]) * seg[2] / 1e6 / self.div


def _lyric_events(tracks: List[_Track]) -> List[Tuple[int, str]]:
    lyric = [(t, _decode(d)) for tr in tracks for t, m, d in tr.events
             if m == 0x05]
    if lyric:
        return sorted(lyric, key=lambda e: e[0])
    best: List[Tuple[int, str]] = []
    for tr in tracks:
        texts = [(t, _decode(d)) for t, m, d in tr.events
                 if m == 0x01 and not _decode(d).startswith("@")]
        if len(texts) > len(best):
            best = texts
    return best


def _header_info(tracks: List[_Track]) -> Dict[str, List[str]]:
    info: Dict[str, List[str]] = {}
    for tr in tracks:
        for _t, m, d in tr.events:
            if m == 0x01:
                s = _decode(d)
                if s.startswith("@") and len(s) > 1:
                    info.setdefault(s[1], []).append(s[2:].strip())
    return info


def _pick_melody(tracks: List[_Track]) -> List[Tuple[int, int, int]]:
    cands = []
    for tr in tracks:
        notes = [(s, e, p) for s, e, p, ch in tr.notes if ch != 9]
        if notes:
            cands.append((tr, notes))
    if not cands:
        return []
    for tr, notes in cands:
        if tr.name.strip().lower() in ("melody", "melodia", "melodía", "vocal",
                                       "voice", "lead"):
            return notes
    top = max(len(n) for _, n in cands)
    pool = [(tr, n) for tr, n in cands if len(n) >= 0.2 * top]
    return max(pool, key=lambda c: sum(p for _, _, p in c[1]) / len(c[1]))[1]


def _monophonic(notes: List[Note]) -> List[Note]:
    out: List[Note] = []
    for n in sorted(notes, key=lambda x: (x.start, -x.pitch)):
        if out and abs(out[-1].start - n.start) < 1e-6:
            continue
        out.append(n)
    return out


def _split_events(events: List[Tuple[int, str]]
                  ) -> List[List[Tuple[int, str]]]:
    """Group (tick, text) into lines using KAR markers and newlines."""
    lines: List[List[Tuple[int, str]]] = [[]]
    for tick, text in events:
        starts_new = text[:1] in ("/", "\\")
        text = text.lstrip("/\\")
        ends_new = text.endswith(("\n", "\r"))
        text = text.replace("\r", "").replace("\n", "")
        if starts_new and lines[-1]:
            lines.append([])
        if text:
            lines[-1].append((tick, text))
        if ends_new and lines[-1]:
            lines.append([])
    return [ln for ln in lines if ln]


def parse_kar(data: bytes, source: Optional[str] = None) -> KarResult:
    division, tracks = _parse_smf(data)
    tempo = _TempoMap(division, tracks)
    info = _header_info(tracks)
    melody = _monophonic([Note(tempo.seconds(s), tempo.seconds(e), p)
                          for s, e, p in _pick_melody(tracks)])

    def note_at(t: float) -> Optional[Note]:
        for n in melody:
            if n.start - 0.05 <= t < n.end:
                return n
        return None

    lines: List[Line] = []
    groups = _split_events(_lyric_events(tracks))
    for gi, grp in enumerate(groups):
        times = [tempo.seconds(t) for t, _ in grp]
        line_end = (tempo.seconds(groups[gi + 1][0][0])
                    if gi + 1 < len(groups) else None)
        syls: List[Syllable] = []
        for i, (_, text) in enumerate(grp):
            nxt = times[i + 1] if i + 1 < len(grp) else line_end
            note = note_at(times[i])
            end = nxt
            if note is not None and (end is None or note.end < end):
                end = note.end
            if end is None:
                end = times[i] + 0.5
            syls.append(Syllable(text=text, start=times[i], end=end,
                                 pitch=note.pitch if note else None))
        syls[-1].text = syls[-1].text.rstrip()
        lines.append(Line(text="".join(s.text for s in syls).strip(),
                          syllables=syls))
    lang = None
    if info.get("L"):
        lang = _LANG.get(info["L"][0].upper())
    title = artist = None
    t_items = info.get("T", [])
    if t_items:
        title = t_items[0]
        artist = t_items[1] if len(t_items) > 1 else None
    return KarResult(lyrics=Lyrics(lines=lines, language=lang, source=source),
                     melody=melody, title=title, artist=artist,
                     info=info.get("I", []))


def read_kar(path) -> KarResult:
    p = Path(path)
    return parse_kar(p.read_bytes(), source=str(p))
