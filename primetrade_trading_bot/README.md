# 🤖 Binance Futures Testnet Trading Bot

> Application submission for **Primetrade.ai** — Python Developer (Trading Bot) role.

A clean, production-style Python CLI bot for placing orders on **Binance Futures Testnet (USDT-M)**. Includes a bonus browser-based visual dashboard.

---

## ✅ Features

| Feature | Details |
|---|---|
| Order types | MARKET, LIMIT, STOP_MARKET |
| Sides | BUY and SELL |
| CLI | Typer + Rich (colored output, validation, help) |
| Logging | Structured file (DEBUG) + console (INFO) |
| Architecture | Separate client / orders / validators / CLI layers |
| Bonus UI | Browser dashboard at `bot/ui/dashboard.html` |

---

## 🛠 Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/primetrade-trading-bot.git
cd primetrade-trading-bot
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get Binance Futures Testnet credentials

1. Go to [https://testnet.binancefuture.com](https://testnet.binancefuture.com)
2. Click **Log In** → authenticate with GitHub
3. Navigate to **API Key** tab → click **Generate**
4. Copy the API Key and Secret (secret shown only once)

### 5. Configure your `.env` file

```bash
cp .env.example .env
```

Open `.env` and paste your credentials:

```
BINANCE_API_KEY=your_actual_testnet_key
BINANCE_API_SECRET=your_actual_testnet_secret
```

---

## 🚀 Running the bot

### Check connectivity

```bash
python cli.py ping
```

Expected output:
```
✓ Testnet is reachable.
```

### Place a MARKET order

```bash
python cli.py place-order --symbol BTCUSDT --side BUY --type MARKET --qty 0.01
```

### Place a LIMIT order

```bash
python cli.py place-order --symbol BTCUSDT --side SELL --type LIMIT --qty 0.01 --price 97000
```

### Place a STOP_MARKET order (bonus order type)

```bash
python cli.py place-order --symbol ETHUSDT --side SELL --type STOP_MARKET --qty 0.1 --stop-price 3000
```

### View account balances

```bash
python cli.py account
```

### Get help

```bash
python cli.py --help
python cli.py place-order --help
```

---

## 🖥 Visual Dashboard (Bonus)

Open `bot/ui/dashboard.html` directly in any browser (no server needed).

- Paste your testnet API key + secret into the credentials panel
- Place orders visually, see live responses in the order table
- Real-time API log panel shows all requests and responses
- Keys never leave your browser — used only to sign requests client-side

---

## 📁 Project Structure

```
primetrade_trading_bot/
├── bot/
│   ├── __init__.py
│   ├── client.py          # Binance REST client + HMAC signing
│   ├── orders.py          # MARKET / LIMIT / STOP_MARKET logic
│   ├── validators.py      # Input validation
│   ├── logging_config.py  # Structured logger (file + console)
│   └── ui/
│       └── dashboard.html # Browser-based visual dashboard (bonus)
├── cli.py                 # Typer CLI entry point
├── logs/                  # Auto-created; sample logs included
│   ├── market_order_sample.log
│   └── limit_order_sample.log
├── .env.example           # Credential template
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 📄 Log Files

Logs are written to `logs/trading_bot_YYYYMMDD.log` and append across runs.

Each entry format:
```
YYYY-MM-DD HH:MM:SS | LEVEL    | module       | message
```

- File handler: `DEBUG` level (all requests, responses, errors)
- Console handler: `INFO` level (order placement events only)

---

## ⚙️ CLI Options Reference

```
place-order
  --symbol   / -s    Trading pair (e.g. BTCUSDT)       [required]
  --side             BUY or SELL                        [required]
  --type     / -t    MARKET | LIMIT | STOP_MARKET       [required]
  --qty      / -q    Order quantity                     [required]
  --price    / -p    Limit price (LIMIT orders only)    [optional]
  --stop-price       Stop price (STOP_MARKET only)      [optional]
  --tif              Time in force: GTC | IOC | FOK     [default: GTC]
```

---

## ⚠️ Assumptions

- Testnet only — base URL is hardcoded to `https://testnet.binancefuture.com`
- Quantity precision must match symbol rules (BTCUSDT minimum: 0.001)
- STOP_MARKET: stop price must be below market for SELL, above for BUY
- API credentials are always loaded from `.env` — never hardcoded
- `requests` is used directly (no `python-binance`) for full control over signing

---

## 📦 Dependencies

```
requests>=2.31.0      # HTTP client
typer[all]>=0.12.0    # CLI framework
rich>=13.7.0          # Terminal formatting
python-dotenv>=1.0.0  # .env loading
```
