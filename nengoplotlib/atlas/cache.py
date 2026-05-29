"""On-disk caching of downloaded atlas data.

Everything ``plot_on_atlas`` fetches from the network (Allen RMA queries,
per-section SVGs, the Swanson polygon file) is small, immutable, and worth
keeping around so that re-running a script -- or running offline -- does not
hit the network again. This module is the single place that knows *where*
those files live and how to fetch-once-then-reuse them.

The cache directory is, in order of precedence:

    $NENGOPLOTLIB_CACHE_DIR          (explicit override)
    $XDG_CACHE_HOME/nengoplotlib/atlas
    ~/.cache/nengoplotlib/atlas      (the usual default)

Network access goes through :func:`urlopen` so tests can monkeypatch a single
function. SSL verification is on by default; set ``NENGOPLOTLIB_INSECURE_SSL=1``
only if you are stuck behind a TLS-intercepting proxy.
"""

from __future__ import annotations

import hashlib
import os
import ssl
import urllib.request
from pathlib import Path
from typing import Optional

_USER_AGENT = "nengoplotlib (https://github.com/ctn-waterloo/nengoplotlib)"
_TIMEOUT = 90


def cache_dir() -> Path:
    """Return (creating if needed) the directory used to cache atlas data."""
    override = os.environ.get("NENGOPLOTLIB_CACHE_DIR")
    if override:
        base = Path(override)
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        root = Path(xdg) if xdg else Path.home() / ".cache"
        base = root / "nengoplotlib" / "atlas"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _ssl_context() -> Optional[ssl.SSLContext]:
    if os.environ.get("NENGOPLOTLIB_INSECURE_SSL") == "1":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    return None  # urlopen uses the default (verifying) context


def urlopen(url: str) -> bytes:
    """Fetch ``url`` and return the raw bytes.

    Isolated as its own function so the whole network layer can be replaced
    with a single ``monkeypatch`` in tests.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=_TIMEOUT, context=_ssl_context()) as resp:
        return resp.read()


def cached_bytes(url: str, *, suffix: str = "", refresh: bool = False) -> bytes:
    """Return the bytes at ``url``, fetching once and caching by URL hash.

    Parameters
    ----------
    url : str
        Resource to fetch.
    suffix : str
        File extension for the on-disk cache entry (cosmetic only; e.g.
        ``".json"`` or ``".svg"`` so the cache is browsable).
    refresh : bool
        If ``True``, bypass any existing cache entry and re-download.
    """
    key = hashlib.sha1(url.encode("utf-8")).hexdigest()
    path = cache_dir() / f"{key}{suffix}"
    if path.exists() and not refresh:
        return path.read_bytes()
    data = urlopen(url)
    # Write atomically so an interrupted download never leaves a truncated
    # file that a later run would treat as a valid cache hit.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return data


def cached_text(url: str, *, suffix: str = "", refresh: bool = False) -> str:
    """:func:`cached_bytes` decoded as UTF-8 (replacing undecodable bytes)."""
    return cached_bytes(url, suffix=suffix, refresh=refresh).decode("utf-8", "replace")
