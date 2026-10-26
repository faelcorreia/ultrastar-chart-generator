import struct
import unittest

from src.readers.kar import parse_kar


def varlen(v):
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    return bytes(reversed(out))


def meta(delta, mtype, payload):
    return varlen(delta) + b"\xff" + bytes([mtype]) + varlen(len(payload)) + payload


def track(events):
    body = b"".join(events) + meta(0, 0x2F, b"")
    return b"MTrk" + struct.pack(">I", len(body)) + body


def smf(tracks, division=480):
    return b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), division) + b"".join(tracks)


def note(delta, pitch, dur, ch=0):
    # note on now (delta), note off after dur (running status avoided)
    return (varlen(delta) + bytes([0x90 | ch, pitch, 90])
            + varlen(dur) + bytes([0x80 | ch, pitch, 0]))


def build(words, enc="utf-8", use_lyric_meta=False):
    """words: list of (text, gap_ticks_before). Each syllable lasts 240 ticks."""
    tempo = track([meta(0, 0x51, (500000).to_bytes(3, "big")),
                   meta(0, 0x01, b"@LPORT"), meta(0, 0x01, b"@TSong Name"),
                   meta(0, 0x01, "@TArtista".encode(enc))])
    kind = 0x05 if use_lyric_meta else 0x01
    mel = [meta(0, 0x03, b"Melody")]
    for i, (text, gap) in enumerate(words):
        mel.append(note(gap, 60 + i, 240))
    ev = [meta(0, 0x03, b"Words")]
    for i, (text, gap) in enumerate(words):
        delta = gap if i == 0 else gap + 240
        ev.append(meta(delta, kind, text.encode(enc)))
    return smf([tempo, track(ev), track(mel)])


class KarTest(unittest.TestCase):
    def test_pt_utf8_syllables_and_melody(self):
        data = build([("/Co", 0), ("ra", 0), ("ção ", 0), ("\\Não ", 240),
                      ("vou ", 0), ("chorar", 0)])
        res = parse_kar(data)
        self.assertEqual(res.lyrics.language, "pt")
        self.assertEqual(res.title, "Song Name")
        self.assertEqual(res.artist, "Artista")
        lines = res.lyrics.lines
        self.assertEqual([l.text for l in lines], ["Coração", "Não vou chorar"])
        syl = lines[0].syllables
        self.assertEqual([s.text for s in syl], ["Co", "ra", "ção"])
        # 480 ticks/beat at 120 bpm: 240 ticks = 0.25 s
        self.assertAlmostEqual(syl[0].start, 0.0)
        self.assertAlmostEqual(syl[1].start, 0.25)
        self.assertAlmostEqual(syl[2].end, 0.75)
        self.assertEqual([s.pitch for s in syl], [60, 61, 62])
        self.assertEqual(len(res.melody), 6)
        self.assertTrue(res.lyrics.has_syllable_timing)

    def test_latin1_fallback_and_lyric_meta(self):
        data = build([("/El ", 0), ("ni", 0), ("ño ", 0), ("don't", 0)],
                     enc="latin-1", use_lyric_meta=True)
        res = parse_kar(data)
        self.assertEqual(res.lyrics.lines[0].text, "El niño don't")

    def test_tempo_change(self):
        tempo = track([meta(0, 0x51, (500000).to_bytes(3, "big")),
                       meta(480, 0x51, (1000000).to_bytes(3, "big"))])
        words = track([meta(0, 0x01, b"a"), meta(960, 0x01, b"b")])
        res = parse_kar(smf([tempo, words]))
        s = res.lyrics.lines[0].syllables
        # tick 960: 0.5 s for first beat + 1 s for second beat
        self.assertAlmostEqual(s[1].start, 1.5)

    def test_newline_in_text(self):
        data = build([("hola\r", 0), ("señor\n", 0), ("adiós", 0)])
        res = parse_kar(data)
        self.assertEqual([l.text for l in res.lyrics.lines],
                         ["hola", "señor", "adiós"])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            parse_kar(b"nope")


if __name__ == "__main__":
    unittest.main()
