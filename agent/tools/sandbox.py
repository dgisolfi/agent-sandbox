"""Sandbox Client

Author(s)
---------
Daniel Nicolas Gisolfi <dgisolfi3@gatech.edu>
"""

import requests


class SandboxClient:
    def __init__(self, base_url: str = "http://sandbox:8000", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health_check(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def run_command(self, **kwargs) -> dict:
        return self._post("/v1/command", kwargs)

    def read_file(self, **kwargs) -> dict:
        return self._post("/v1/files/read", kwargs)

    def write_file(self, **kwargs) -> dict:
        return self._post("/v1/files/write", kwargs)

    def list_dir(self, **kwargs) -> dict:
        return self._post("/v1/files/list", kwargs)

    def _post(self, path: str, payload: dict) -> dict:
        try:
            response = requests.post(
                f"{self.base_url}{path}", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            envelope = response.json()
            if envelope.get("ok"):
                return {"ok": True, **(envelope.get("data") or {})}
            error = envelope.get("error") or {}
            return {
                "ok": False,
                "error": error.get("message", "Sandbox request failed"),
                "error_type": error.get("type", "sandbox_error"),
                "details": error.get("details", {}),
            }
        except requests.RequestException as exc:
            return {"ok": False, "error": str(exc), "error_type": "connection_error"}
