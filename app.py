"""
Degeneracy Capital — Paper Trading Simulator
FastAPI + SQLite + yfinance + Black-Scholes
"""

import os
import json
import math
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, date
from typing import Optional

import httpx
import yfinance as yf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH = os.environ.get("DB_PATH", "degcap.db")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
PRICE_POLL_INTERVAL = 15 * 60  # 15 minutes

DEFAULT_PLAYERS = ["Chungus", "Fungus", "Nasseem", "Gremlin"]

WATCHLIST = [
    "FRO", "DHT", "NAT", "XLE", "XOM", "RTX", "GLD",
    "SONY", "MSI", "AXON", "AMBA", "NVDA", "NTR", "MOS",
    "CF", "APD", "LIN", "AAL", "DAL", "BTC-USD", "ETH-USD",
]

STARTING_CASH = 1_000_000.0
RISK_FREE_RATE = 0.05
DEFAULT_IV = 0.30


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def db():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS players (
                id   INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                cash REAL DEFAULT 1000000.0
            );

            CREATE TABLE IF NOT EXISTS positions (
                id            INTEGER PRIMARY KEY,
                player_id     INTEGER,
                ticker        TEXT,
                quantity      REAL,
                avg_cost      REAL,
                position_type TEXT,
                strike        REAL,
                expiry        TEXT
            );

            CREATE TABLE IF NOT EXISTS trades (
                id        INTEGER PRIMARY KEY,
                player_id INTEGER,
                ticker    TEXT,
                action    TEXT,
                quantity  REAL,
                price     REAL,
                timestamp TEXT
            );

            CREATE TABLE IF NOT EXISTS price_cache (
                ticker     TEXT PRIMARY KEY,
                price      REAL,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id        INTEGER PRIMARY KEY,
                player    TEXT,
                value     REAL,
                cash      REAL,
                pnl       REAL,
                ts        TEXT
            );
        """)

        # Seed default players
        for name in DEFAULT_PLAYERS:
            conn.execute(
                "INSERT OR IGNORE INTO players (name, cash) VALUES (?, ?)",
                (name, STARTING_CASH),
            )


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------

_price_lock = threading.Lock()


def fetch_price(ticker: str) -> float:
    """Fetch live price via yfinance. Returns 0 on failure."""
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        price = getattr(info, "last_price", None) or getattr(info, "regular_market_price", None)
        if price:
            return float(price)
        hist = t.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
        return 0.0
    except Exception:
        return 0.0


def get_cached_price(ticker: str) -> float:
    with db() as conn:
        row = conn.execute(
            "SELECT price, updated_at FROM price_cache WHERE ticker = ?", (ticker,)
        ).fetchone()
    if row:
        return float(row["price"])
    # Not cached — fetch now
    price = fetch_price(ticker)
    if price > 0:
        store_price(ticker, price)
    return price


def store_price(ticker: str, price: float):
    now = datetime.utcnow().isoformat()
    with db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO price_cache (ticker, price, updated_at) VALUES (?, ?, ?)",
            (ticker, price, now),
        )


def refresh_all_prices():
    """Refresh prices for all watchlist tickers."""
    for ticker in WATCHLIST:
        price = fetch_price(ticker)
        if price > 0:
            store_price(ticker, price)
    # Also refresh any tickers in active positions not on watchlist
    with db() as conn:
        rows = conn.execute("SELECT DISTINCT ticker FROM positions").fetchall()
    extra = {r["ticker"] for r in rows} - set(WATCHLIST)
    for ticker in extra:
        price = fetch_price(ticker)
        if price > 0:
            store_price(ticker, price)


def take_portfolio_snapshots():
    with db() as conn:
        players = [r["name"] for r in conn.execute("SELECT name FROM players").fetchall()]
    now = datetime.utcnow().isoformat()
    for name in players:
        try:
            p = compute_portfolio(name)
            if p:
                with db() as conn:
                    conn.execute(
                        "INSERT INTO portfolio_snapshots (player, value, cash, pnl, ts) VALUES (?,?,?,?,?)",
                        (name, p["total_value"], p["cash"], p["total_pnl"], now)
                    )
        except Exception as e:
            print(f"[snapshot] {name}: {e}")


def price_poller():
    """Background thread: refresh prices + snapshots every PRICE_POLL_INTERVAL seconds."""
    while True:
        try:
            refresh_all_prices()
            take_portfolio_snapshots()
        except Exception as e:
            print(f"[price_poller] error: {e}")
        time.sleep(PRICE_POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Black-Scholes options pricing
# ---------------------------------------------------------------------------

def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> float:
    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def option_premium(
    ticker: str,
    strike: float,
    expiry: str,          # ISO date string YYYY-MM-DD
    option_type: str,     # 'call' or 'put'
    quantity: float = 1.0,
) -> float:
    """Return total premium for quantity contracts (1 contract = 100 shares)."""
    S = get_cached_price(ticker)
    if S <= 0:
        raise ValueError(f"Could not get price for {ticker}")
    K = strike
    expiry_date = date.fromisoformat(expiry)
    today = date.today()
    T = max((expiry_date - today).days / 365.0, 0.0)
    per_share = black_scholes(S, K, T, RISK_FREE_RATE, DEFAULT_IV, option_type)
    return per_share * 100 * quantity  # 1 contract = 100 shares


def option_current_value(position: dict) -> float:
    """Mark-to-market value of an options position."""
    ticker = position["ticker"]
    strike = position["strike"]
    expiry = position["expiry"]
    option_type = position["position_type"]  # 'call' or 'put'
    quantity = position["quantity"]
    S = get_cached_price(ticker)
    if S <= 0 or not strike or not expiry:
        return 0.0
    expiry_date = date.fromisoformat(expiry)
    today = date.today()
    T = max((expiry_date - today).days / 365.0, 0.0)
    per_share = black_scholes(S, strike, T, RISK_FREE_RATE, DEFAULT_IV, option_type)
    return per_share * 100 * quantity


# ---------------------------------------------------------------------------
# Portfolio valuation
# ---------------------------------------------------------------------------

def get_player_id(conn, name: str) -> Optional[int]:
    row = conn.execute("SELECT id FROM players WHERE name = ?", (name,)).fetchone()
    return row["id"] if row else None


def get_or_create_player(conn, name: str) -> int:
    row = conn.execute("SELECT id FROM players WHERE name = ?", (name,)).fetchone()
    if row:
        return row["id"]
    conn.execute("INSERT INTO players (name, cash) VALUES (?, ?)", (name, STARTING_CASH))
    return conn.execute("SELECT id FROM players WHERE name = ?", (name,)).fetchone()["id"]


def compute_portfolio(player_name: str) -> dict:
    with db() as conn:
        player_row = conn.execute(
            "SELECT id, name, cash FROM players WHERE name = ?", (player_name,)
        ).fetchone()
        if not player_row:
            return None

        player_id = player_row["id"]
        cash = float(player_row["cash"])

        positions_raw = conn.execute(
            "SELECT * FROM positions WHERE player_id = ? AND quantity != 0",
            (player_id,),
        ).fetchall()

        trades_raw = conn.execute(
            "SELECT * FROM trades WHERE player_id = ? ORDER BY timestamp DESC LIMIT 50",
            (player_id,),
        ).fetchall()

    positions = []
    total_market_value = 0.0
    total_cost_basis = 0.0

    for pos in positions_raw:
        pos = dict(pos)
        ticker = pos["ticker"]
        qty = float(pos["quantity"])
        avg_cost = float(pos["avg_cost"])
        ptype = pos["position_type"]

        current_price = get_cached_price(ticker)

        if ptype in ("call", "put"):
            cost_basis = avg_cost  # total premium paid stored
            market_value = option_current_value(pos)
            pnl = market_value - cost_basis
            pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0.0
        elif ptype == "short":
            # Shorted: we received avg_cost per share, now must buy back at current_price
            # P&L = (avg_cost - current_price) * qty
            cost_basis = avg_cost * qty  # proceeds received
            market_value = current_price * qty  # cost to cover
            pnl = (avg_cost - current_price) * qty
            pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0.0
            # For net worth: short positions reduce total value by current_price*qty
            # We add back cost_basis (collateral already deducted) and subtract cover cost
        else:
            # Long
            cost_basis = avg_cost * qty
            market_value = current_price * qty
            pnl = market_value - cost_basis
            pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0.0

        positions.append({
            "ticker": ticker,
            "quantity": qty,
            "avg_cost": avg_cost,
            "position_type": ptype,
            "strike": pos.get("strike"),
            "expiry": pos.get("expiry"),
            "current_price": current_price,
            "cost_basis": round(cost_basis, 2),
            "market_value": round(market_value, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
        })

        if ptype == "long":
            total_market_value += market_value
            total_cost_basis += cost_basis
        elif ptype in ("call", "put"):
            total_market_value += market_value
            total_cost_basis += cost_basis
        elif ptype == "short":
            # Short: net effect on portfolio = proceeds received - cost to cover
            # already deducted from cash when shorting; PnL accumulates
            total_market_value += pnl  # net gain/loss from short

    total_value = cash + total_market_value
    total_pnl = total_value - STARTING_CASH
    total_pnl_pct = (total_pnl / STARTING_CASH) * 100

    trades = [dict(t) for t in trades_raw]

    return {
        "player": player_name,
        "cash": round(cash, 2),
        "positions": positions,
        "total_market_value": round(total_market_value, 2),
        "total_value": round(total_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "trades": trades,
    }


# ---------------------------------------------------------------------------
# Discord webhook
# ---------------------------------------------------------------------------

def post_discord(message: str):
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        httpx.post(
            DISCORD_WEBHOOK_URL,
            json={"content": message},
            timeout=5,
        )
    except Exception as e:
        print(f"[discord] webhook error: {e}")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Degeneracy Capital")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def on_startup():
    init_db()
    # Seed initial prices in background so startup is fast
    t = threading.Thread(target=refresh_all_prices, daemon=True)
    t.start()
    # Start background price poller
    poller = threading.Thread(target=price_poller, daemon=True)
    poller.start()


# ---------------------------------------------------------------------------
# Frontend routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return f.read()


@app.get("/portfolio/{player}", response_class=HTMLResponse)
async def portfolio_page(player: str):
    with open("static/portfolio.html") as f:
        return f.read()


@app.get("/trade", response_class=HTMLResponse)
async def trade_page():
    with open("static/trade.html") as f:
        return f.read()


@app.get("/prices", response_class=HTMLResponse)
async def prices_page():
    with open("static/prices.html") as f:
        return f.read()


# ---------------------------------------------------------------------------
# API — prices
# ---------------------------------------------------------------------------

@app.get("/api/price/{ticker}")
async def api_price(ticker: str):
    price = get_cached_price(ticker.upper())
    if price <= 0:
        raise HTTPException(status_code=404, detail=f"No price available for {ticker}")
    return {"ticker": ticker.upper(), "price": price}


@app.get("/api/prices")
async def api_prices():
    results = {}
    with db() as conn:
        rows = conn.execute("SELECT ticker, price, updated_at FROM price_cache").fetchall()
    for row in rows:
        results[row["ticker"]] = {"price": row["price"], "updated_at": row["updated_at"]}
    return results


# ---------------------------------------------------------------------------
# API — portfolio / leaderboard
# ---------------------------------------------------------------------------

@app.get("/api/portfolio/{player}")
async def api_portfolio(player: str):
    data = compute_portfolio(player)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Player '{player}' not found")
    return data


@app.get("/api/leaderboard")
async def api_leaderboard():
    with db() as conn:
        rows = conn.execute("SELECT name FROM players ORDER BY name").fetchall()
    players = [r["name"] for r in rows]
    results = []
    for name in players:
        p = compute_portfolio(name)
        if p:
            results.append({
                "player": p["player"],
                "total_value": p["total_value"],
                "cash": p["cash"],
                "total_pnl": p["total_pnl"],
                "total_pnl_pct": p["total_pnl_pct"],
                "position_count": len(p["positions"]),
            })
    results.sort(key=lambda x: x["total_value"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results


# ---------------------------------------------------------------------------
# API — trading
# ---------------------------------------------------------------------------

class TradeRequest(BaseModel):
    player: str
    ticker: str
    action: str          # buy | sell | short | cover | call | put
    quantity: float
    strike: Optional[float] = None
    expiry: Optional[str] = None   # ISO date YYYY-MM-DD


@app.post("/api/trade")
async def api_trade(req: TradeRequest):
    player = req.player.strip()
    ticker = req.ticker.strip().upper()
    action = req.action.strip().lower()
    quantity = req.quantity
    strike = req.strike
    expiry = req.expiry

    if quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be positive")

    valid_actions = {"buy", "sell", "short", "cover", "call", "put"}
    if action not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Invalid action. Use: {valid_actions}")

    if action in ("call", "put"):
        if not strike or not expiry:
            raise HTTPException(status_code=400, detail="Options require strike and expiry")
        try:
            date.fromisoformat(expiry)
        except ValueError:
            raise HTTPException(status_code=400, detail="Expiry must be YYYY-MM-DD")

    current_price = get_cached_price(ticker)
    if current_price <= 0:
        # Try fetching live
        current_price = fetch_price(ticker)
        if current_price <= 0:
            raise HTTPException(status_code=400, detail=f"Cannot get price for {ticker}")
        store_price(ticker, current_price)

    with db() as conn:
        player_id = get_or_create_player(conn, player)
        cash_row = conn.execute("SELECT cash FROM players WHERE id = ?", (player_id,)).fetchone()
        cash = float(cash_row["cash"])

        now = datetime.utcnow().isoformat()

        if action == "buy":
            cost = current_price * quantity
            if cost > cash:
                raise HTTPException(status_code=400, detail=f"Insufficient cash. Need ${cost:,.2f}, have ${cash:,.2f}")

            # Update position
            existing = conn.execute(
                "SELECT id, quantity, avg_cost FROM positions WHERE player_id=? AND ticker=? AND position_type='long'",
                (player_id, ticker),
            ).fetchone()

            if existing:
                old_qty = float(existing["quantity"])
                old_cost = float(existing["avg_cost"])
                new_qty = old_qty + quantity
                new_avg = (old_qty * old_cost + quantity * current_price) / new_qty
                conn.execute(
                    "UPDATE positions SET quantity=?, avg_cost=? WHERE id=?",
                    (new_qty, new_avg, existing["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO positions (player_id, ticker, quantity, avg_cost, position_type) VALUES (?, ?, ?, ?, 'long')",
                    (player_id, ticker, quantity, current_price),
                )

            conn.execute("UPDATE players SET cash=? WHERE id=?", (cash - cost, player_id))
            conn.execute(
                "INSERT INTO trades (player_id, ticker, action, quantity, price, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (player_id, ticker, action, quantity, current_price, now),
            )

        elif action == "sell":
            existing = conn.execute(
                "SELECT id, quantity FROM positions WHERE player_id=? AND ticker=? AND position_type='long'",
                (player_id, ticker),
            ).fetchone()
            if not existing or float(existing["quantity"]) < quantity:
                held = float(existing["quantity"]) if existing else 0
                raise HTTPException(status_code=400, detail=f"Insufficient long position. Holding {held} shares")

            proceeds = current_price * quantity
            new_qty = float(existing["quantity"]) - quantity
            if new_qty == 0:
                conn.execute("DELETE FROM positions WHERE id=?", (existing["id"],))
            else:
                conn.execute("UPDATE positions SET quantity=? WHERE id=?", (new_qty, existing["id"]))

            conn.execute("UPDATE players SET cash=? WHERE id=?", (cash + proceeds, player_id))
            conn.execute(
                "INSERT INTO trades (player_id, ticker, action, quantity, price, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (player_id, ticker, action, quantity, current_price, now),
            )

        elif action == "short":
            # Borrow and sell — receive proceeds, owe the shares
            # Require 150% collateral (50% margin)
            proceeds = current_price * quantity
            collateral = proceeds * 1.5
            if collateral > cash:
                raise HTTPException(status_code=400, detail=f"Insufficient margin. Need ${collateral:,.2f} (150% collateral), have ${cash:,.2f}")

            existing = conn.execute(
                "SELECT id, quantity, avg_cost FROM positions WHERE player_id=? AND ticker=? AND position_type='short'",
                (player_id, ticker),
            ).fetchone()

            if existing:
                old_qty = float(existing["quantity"])
                old_cost = float(existing["avg_cost"])
                new_qty = old_qty + quantity
                new_avg = (old_qty * old_cost + quantity * current_price) / new_qty
                conn.execute(
                    "UPDATE positions SET quantity=?, avg_cost=? WHERE id=?",
                    (new_qty, new_avg, existing["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO positions (player_id, ticker, quantity, avg_cost, position_type) VALUES (?, ?, ?, ?, 'short')",
                    (player_id, ticker, quantity, current_price),
                )

            # Deduct collateral from cash, add proceeds
            conn.execute("UPDATE players SET cash=? WHERE id=?", (cash + proceeds - collateral, player_id))
            conn.execute(
                "INSERT INTO trades (player_id, ticker, action, quantity, price, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (player_id, ticker, action, quantity, current_price, now),
            )

        elif action == "cover":
            # Buy back shorted shares
            existing = conn.execute(
                "SELECT id, quantity, avg_cost FROM positions WHERE player_id=? AND ticker=? AND position_type='short'",
                (player_id, ticker),
            ).fetchone()
            if not existing or float(existing["quantity"]) < quantity:
                held = float(existing["quantity"]) if existing else 0
                raise HTTPException(status_code=400, detail=f"Insufficient short position. Short {held} shares")

            cover_cost = current_price * quantity
            short_qty = float(existing["quantity"])
            short_avg = float(existing["avg_cost"])

            # Collateral to return: 1.5x * avg_cost * quantity
            collateral_return = short_avg * quantity * 1.5
            # PnL: (short_price - cover_price) * qty
            pnl = (short_avg - current_price) * quantity

            # Net cash effect: return collateral, account for pnl
            # cash was reduced by collateral_return - proceeds = collateral - avg*qty
            # on cover: give back (current_price * qty) to buy shares, return collateral
            cash_return = collateral_return - cover_cost

            new_qty = short_qty - quantity
            if new_qty == 0:
                conn.execute("DELETE FROM positions WHERE id=?", (existing["id"],))
            else:
                conn.execute("UPDATE positions SET quantity=? WHERE id=?", (new_qty, existing["id"]))

            conn.execute("UPDATE players SET cash=? WHERE id=?", (cash + cash_return, player_id))
            conn.execute(
                "INSERT INTO trades (player_id, ticker, action, quantity, price, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (player_id, ticker, action, quantity, current_price, now),
            )

        elif action in ("call", "put"):
            # Buy an option contract
            try:
                premium = option_premium(ticker, strike, expiry, action, quantity)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

            if premium > cash:
                raise HTTPException(status_code=400, detail=f"Insufficient cash. Premium ${premium:,.2f}, have ${cash:,.2f}")

            existing = conn.execute(
                "SELECT id, quantity, avg_cost FROM positions WHERE player_id=? AND ticker=? AND position_type=? AND strike=? AND expiry=?",
                (player_id, ticker, action, strike, expiry),
            ).fetchone()

            if existing:
                old_qty = float(existing["quantity"])
                old_cost = float(existing["avg_cost"])
                new_qty = old_qty + quantity
                new_avg_cost = old_cost + premium  # total premium paid
                conn.execute(
                    "UPDATE positions SET quantity=?, avg_cost=? WHERE id=?",
                    (new_qty, new_avg_cost, existing["id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO positions (player_id, ticker, quantity, avg_cost, position_type, strike, expiry) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (player_id, ticker, quantity, premium, action, strike, expiry),
                )

            conn.execute("UPDATE players SET cash=? WHERE id=?", (cash - premium, player_id))
            conn.execute(
                "INSERT INTO trades (player_id, ticker, action, quantity, price, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (player_id, ticker, action, quantity, current_price, now),
            )

    # Discord notification
    action_emoji = {
        "buy": "📈", "sell": "📉", "short": "🐻", "cover": "🤝",
        "call": "☎️", "put": "🪃",
    }.get(action, "💸")
    msg = (
        f"{action_emoji} **{player}** {action.upper()} {quantity:g}x **{ticker}** @ ${current_price:,.2f}"
    )
    if action in ("call", "put"):
        msg += f" | Strike ${strike:,.2f} | Expiry {expiry}"
    post_discord(msg)

    return {
        "status": "ok",
        "player": player,
        "ticker": ticker,
        "action": action,
        "quantity": quantity,
        "price": current_price,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# API — admin / utility
# ---------------------------------------------------------------------------

@app.get("/charts", response_class=HTMLResponse)
async def charts_page():
    with open("static/charts.html") as f:
        return f.read()


@app.get("/api/history/{player}")
async def api_history(player: str):
    with db() as conn:
        rows = conn.execute(
            "SELECT ts, value, cash, pnl FROM portfolio_snapshots WHERE player=? ORDER BY ts ASC LIMIT 500",
            (player,)
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/history")
async def api_history_all():
    with db() as conn:
        rows = conn.execute(
            "SELECT player, ts, value, pnl FROM portfolio_snapshots ORDER BY ts ASC LIMIT 2000"
        ).fetchall()
    result = {}
    for r in rows:
        result.setdefault(r["player"], []).append({"ts": r["ts"], "value": r["value"], "pnl": r["pnl"]})
    return result


@app.get("/api/watchlist")
async def api_watchlist():
    return {"tickers": WATCHLIST}


@app.post("/api/refresh_prices")
async def api_refresh_prices():
    """Manually trigger a price refresh."""
    t = threading.Thread(target=refresh_all_prices, daemon=True)
    t.start()
    return {"status": "refresh triggered"}


@app.get("/api/players")
async def api_players():
    with db() as conn:
        rows = conn.execute("SELECT name FROM players ORDER BY name").fetchall()
    return {"players": [r["name"] for r in rows]}
