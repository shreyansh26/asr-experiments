from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


def reset_prefix_cache_url(base_url: str) -> str:
    parsed_url = urlsplit(base_url)
    base_path = parsed_url.path.rstrip("/")
    if base_path == "/v1":
        base_path = ""
    elif base_path.endswith("/v1"):
        base_path = base_path[:-3]

    reset_path = f"{base_path}/reset_prefix_cache" if base_path else "/reset_prefix_cache"
    return urlunsplit(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            reset_path,
            "",
            "",
        )
    )


def reset_prefix_cache(base_url: str, timeout_seconds: float = 10.0) -> str:
    endpoint = reset_prefix_cache_url(base_url)
    request = Request(endpoint, method="POST")
    try:
        with urlopen(request, timeout=timeout_seconds):
            return endpoint
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        message = f"Failed to reset prefix cache at {endpoint}: HTTP {exc.code}"
        if detail:
            message = f"{message} {detail}"
        raise RuntimeError(message) from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reset prefix cache at {endpoint}: {exc}") from exc
