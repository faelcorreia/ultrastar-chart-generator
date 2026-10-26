"""Minimal injectable HTTP layer (stdlib only).

Clients take `http` as a callable `http(url, headers) -> HttpResponse | None`.
Tests pass a fake; production uses `urllib_get`.
"""
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class HttpResponse:
    status: int
    body: bytes = b""

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))


HttpGet = Callable[[str, Dict[str, str]], Optional[HttpResponse]]


def urllib_get(url: str, headers: Optional[Dict[str, str]] = None,
               timeout: float = 15.0) -> Optional[HttpResponse]:
    """GET following redirects. HTTP errors become responses; network errors None."""
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return HttpResponse(resp.status, resp.read())
    except urllib.error.HTTPError as err:
        return HttpResponse(err.code, err.read() or b"")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
