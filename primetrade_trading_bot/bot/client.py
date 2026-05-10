import hashlib
import hmac
import time
from typing import Any, Dict, Optional
from urllib.parse import urlencode

import requests

from bot.logging_config import setup_logger

logger = setup_logger("client")

BASE_URL = "https://testnet.binancefuture.com"
TIMEOUT = 10


class BinanceAPIError(Exception):
    """Raised when Binance returns a non-2xx response."""
    def __init__(self, code: Any, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class BinanceClient:
    """Thin REST wrapper for Binance Futures Testnet."""

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

    # ── signing ──────────────────────────────────────────────────────────
    def _sign(self, params: Dict[str, Any]) -> str:
        qs = urlencode(params)
        return hmac.new(self.api_secret, qs.encode(), hashlib.sha256).hexdigest()

    @staticmethod
    def _ts() -> int:
        return int(time.time() * 1000)

    # ── HTTP helpers ──────────────────────────────────────────────────────
    def _get(self, path: str, params: Optional[Dict] = None, signed: bool = False) -> Dict:
        params = dict(params or {})
        if signed:
            params["timestamp"] = self._ts()
            params["signature"] = self._sign(params)
        url = BASE_URL + path
        logger.debug("GET %s params=%s", url, params)
        try:
            r = self.session.get(url, params=params, timeout=TIMEOUT)
        except requests.exceptions.RequestException as exc:
            logger.error("Network error GET %s: %s", url, exc)
            raise
        return self._parse(r)

    def _post(self, path: str, params: Dict, signed: bool = True) -> Dict:
        params = dict(params)
        if signed:
            params["timestamp"] = self._ts()
            params["signature"] = self._sign(params)
        url = BASE_URL + path
        logger.debug(
            "POST %s payload=%s", url,
            {k: v for k, v in params.items() if k != "signature"}
        )
        try:
            r = self.session.post(url, data=params, timeout=TIMEOUT)
        except requests.exceptions.RequestException as exc:
            logger.error("Network error POST %s: %s", url, exc)
            raise
        return self._parse(r)

    def _parse(self, r: requests.Response) -> Dict:
        logger.debug("Response [%s]: %s", r.status_code, r.text[:600])
        if not r.ok:
            try:
                err = r.json()
                code, msg = err.get("code", r.status_code), err.get("msg", r.text)
            except Exception:
                code, msg = r.status_code, r.text
            logger.error("API error %s: %s", code, msg)
            raise BinanceAPIError(code=code, message=msg)
        return r.json()

    # ── public endpoints ──────────────────────────────────────────────────
    def ping(self) -> bool:
        self._get("/fapi/v1/ping")
        return True

    def get_server_time(self) -> int:
        return self._get("/fapi/v1/time").get("serverTime", 0)

    def get_exchange_info(self) -> Dict:
        return self._get("/fapi/v1/exchangeInfo")

    def get_ticker(self, symbol: str) -> Dict:
        return self._get("/fapi/v1/ticker/price", params={"symbol": symbol})

    def get_account(self) -> Dict:
        return self._get("/fapi/v2/account", signed=True)

    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._get("/fapi/v1/openOrders", params=params, signed=True)

    # ── order endpoint ────────────────────────────────────────────────────
    def place_order(self, **kwargs) -> Dict:
        return self._post("/fapi/v1/order", params=kwargs)
