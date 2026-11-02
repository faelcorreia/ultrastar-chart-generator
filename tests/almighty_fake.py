"""listen falso para os testes do K2b: devolve transcript.json v2 e alignment.json no contrato do
almighty-listener, com palavras espalhadas igualmente na janela de cada linha (tempos absolutos).

`fail_on`: primeiras palavras de linhas que devem "falhar" (entram em failed_lines).
`mode`: "ok", "v1" (transcript sem linhas) ou "short" (uma palavra a menos no alignment).
"""
import json
from pathlib import Path


def split_syllables(word):
    """Divisão fictícia tipo pyphen: duas sílabas em palavras longas."""
    return [word[:2], word[2:]] if len(word) > 3 else [word]


class FakeResult:
    def __init__(self, files):
        self._files = files

    def get(self, artifact):
        if artifact not in self._files:
            raise KeyError(f"artefato {artifact!r} não foi produzido")
        return self._files[artifact]


class FakeAlmighty:
    def __init__(self, root, fail_on=(), mode="ok", error=None):
        """`root`: pasta do teste onde os JSONs falsos são gravados."""
        self.fail_on = set(fail_on)
        self.mode = mode
        self.error = error
        self.calls = []
        self.root = Path(root)

    def __call__(self, path, want=None, *, device="cpu", options=None,
                 cache_dir=None, work_dir=None):
        self.calls.append({"path": Path(path), "want": list(want), "options": options,
                           "cache_dir": Path(cache_dir), "work_dir": Path(work_dir)})
        if self.error is not None:
            raise self.error
        opt = options["transcript"]
        language = opt["language"]
        t_words, a_words, failed = [], [], []
        for k, line in enumerate(opt["lines"]):
            words = line["text"].split()
            s, e = line["window"] if line["window"] is not None else (0.0, float(len(words)))
            if words[0] in self.fail_on:
                failed.append(k)
            step = (e - s) / len(words)
            for j, word in enumerate(words):
                ws, we = s + j * step, s + (j + 1) * step
                t_words.append({"text": word, "start": ws, "end": we, "line": k,
                                "syllables": []})
                parts = split_syllables(word)
                sub = (we - ws) / len(parts)
                a_words.append({
                    "text": word, "start": ws, "end": we,
                    "syllables": [{"text": p, "start": ws + i * sub, "end": ws + (i + 1) * sub,
                                   "phonemes": []} for i, p in enumerate(parts)],
                })
        if self.mode == "short":
            a_words = a_words[:-1]
        transcript = {"schema_version": 1 if self.mode == "v1" else 2, "language": language,
                      "words": t_words}
        if self.mode != "v1":
            transcript["failed_lines"] = failed
        alignment = {"schema_version": 1, "language": language, "words": a_words}
        t_path = self.root / "transcript.json"
        a_path = self.root / "alignment.json"
        t_path.write_text(json.dumps(transcript), encoding="utf-8")
        a_path.write_text(json.dumps(alignment), encoding="utf-8")
        return FakeResult({"transcript": t_path, "alignment": a_path})
