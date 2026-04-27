#!/usr/bin/env python3
"""
Degeneracy Capital — Exit Signal Monitor (LLM-enhanced)
Polls RSS feeds, classifies signals via Haiku on OpenRouter, posts to Discord.
"""

import os
import json
import time
import hashlib
import logging
import xml.etree.ElementTree as ET
from urllib.request import urlopen, Request as URLRequest

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

DISCORD_WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
SEEN_FILE = os.path.expanduser("~/.degcap_seen.json")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "900"))
MODEL = "anthropic/claude-haiku-4-5"

THESES = [
    {
        "id": "HORMUZ",
        "tickers": "FRO / DHT / NAT / RTX / GLD",
        "description": (
            "Long tankers and defense on Strait of Hormuz disruption. "
            "Exit: Iran nuclear deal, US-Iran ceasefire, strait reopened, sanctions lifted. "
            "Watch: diplomatic talks, de-escalation, shipping insurance normalising."
        ),
        "feeds": [
            "https://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
            "https://www.aljazeera.com/xml/rss/all.xml",
        ],
        "keywords": ["hormuz", "iran", "strait", "tanker", "persian gulf", "nuclear deal", "sanctions", "ceasefire"],
    },
    {
        "id": "FERTILIZER",
        "tickers": "NTR / MOS / CF",
        "description": (
            "Long fertilizer on Russia/Belarus potash supply gap. "
            "Exit: sanctions lifted, potash glut, Ukraine peace restoring exports. Watch: ceasefire talks."
        ),
        "feeds": ["https://feeds.reuters.com/reuters/businessNews"],
        "keywords": ["potash", "fertilizer", "russia sanctions", "belarus", "nutrien", "mosaic", "nitrogen", "ukraine ceasefire"],
    },
    {
        "id": "SURVEILLANCE/NVDA",
        "tickers": "NVDA / AMBA / AXON / MSI / SONY",
        "description": (
            "Long AI surveillance and NVIDIA compute. "
            "Exit: NVDA revenue miss, guidance cut, export controls expanded, hyperscaler custom silicon reducing NVIDIA orders, capex cuts. "
            "Watch: early signs of above."
        ),
        "feeds": ["https://feeds.reuters.com/reuters/technologyNews"],
        "keywords": ["nvidia", "export controls", "chip ban", "data center", "ai chip", "hyperscaler", "google tpu", "aws trainium", "capex", "earnings"],
    },
    {
        "id": "AIRLINE_SHORT",
        "tickers": "AAL / DAL (shorts — cover signal)",
        "description": (
            "Short airlines on fuel exposure. Cover: oil below $80, airlines announce fuel hedging, Hormuz resolved. Watch: oil stabilising."
        ),
        "feeds": ["https://feeds.reuters.com/reuters/topNews"],
        "keywords": ["oil price", "crude", "fuel", "american airlines", "delta airlines", "airline hedge", "wti", "brent"],
    },
]

SYSTEM_PROMPT = """You are a financial signal classifier for an event-driven portfolio.
Given a news headline and thesis, classify the signal.
Respond ONLY with valid JSON: {"signal": "EXIT_NOW"|"CONSIDER_EXIT"|"WATCH"|"NONE", "reason": "<one sentence>"}
EXIT_NOW: strong evidence thesis is broken. CONSIDER_EXIT: meaningful movement toward thesis break. WATCH: tangentially relevant. NONE: not relevant."""

def load_seen():
    try:
        with open(SEEN_FILE) as f:
            return set(json.load(f))
    except Exception:
        return set()

def save_seen(seen):
    try:
        with open(SEEN_FILE, "w") as f:
            json.dump(list(seen), f)
    except Exception as e:
        log.error(f"save_seen: {e}")

def fetch_feed(url):
    try:
        req = URLRequest(url, headers={"User-Agent": "DegeneracyCapital/1.0"})
        with urlopen(req, timeout=15) as r:
            content = r.read()
        root = ET.fromstring(content)
        items = []
        for item in root.findall(".//item"):
            items.append({
                "title": (item.findtext("title") or "").strip(),
                "link": (item.findtext("link") or "").strip(),
                "desc": (item.findtext("description") or "").strip(),
            })
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall(".//atom:entry", ns):
            link_el = entry.find("atom:link", ns)
            items.append({
                "title": (entry.findtext("atom:title", namespaces=ns) or "").strip(),
                "link": (link_el.get("href", "") if link_el is not None else "").strip(),
                "desc": (entry.findtext("atom:summary", namespaces=ns) or "").strip(),
            })
        return items
    except Exception as e:
        log.warning(f"Feed error {url}: {e}")
        return []

def is_relevant(item, thesis):
    text = (item["title"] + " " + item["desc"]).lower()
    return any(kw in text for kw in thesis["keywords"])

def classify_with_llm(headline, desc, thesis):
    if not OPENROUTER_KEY:
        return keyword_fallback(headline + " " + desc)

    user_msg = f"Thesis: {thesis['description']}\nTickers: {thesis['tickers']}\n\nHeadline: {headline}\nSummary: {desc[:300]}"
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 80,
        "temperature": 0.1,
    }).encode()

    try:
        req = URLRequest(
            "https://openrouter.ai/api/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "HTTP-Referer": "https://degeneracy.capital",
                "X-Title": "Degeneracy Capital",
            },
            method="POST",
        )
        with urlopen(req, timeout=20) as r:
            resp = json.loads(r.read())
        content = resp["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
        return result.get("signal", "NONE"), result.get("reason", "")
    except Exception as e:
        log.warning(f"LLM error: {e}, falling back to keyword")
        return keyword_fallback(headline + " " + desc)

def keyword_fallback(text):
    text = text.lower()
    for t in ["deal signed", "agreement reached", "ceasefire", "sanctions lifted", "guidance cut", "revenue miss"]:
        if t in text:
            return "CONSIDER_EXIT", f"keyword: '{t}'"
    return "WATCH", "keyword relevance"

EMOJI = {"EXIT_NOW": "🔴", "CONSIDER_EXIT": "🟡", "WATCH": "🔵"}

def post_discord(signal, thesis_id, tickers, headline, link, reason):
    emoji = EMOJI.get(signal, "⚪")
    content = f"{emoji} **{signal}** — {thesis_id}\n**Tickers:** {tickers}\n**Signal:** {reason}\n**Headline:** {headline}\n{link}"
    log.info(f"[{signal}] {thesis_id} — {headline[:80]}")
    if not DISCORD_WEBHOOK:
        return
    try:
        req = URLRequest(DISCORD_WEBHOOK, data=json.dumps({"content": content}).encode(),
                         headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=10):
            pass
    except Exception as e:
        log.error(f"Discord error: {e}")

def run_once(seen):
    alerts = 0
    for thesis in THESES:
        items = []
        for url in thesis["feeds"]:
            items.extend(fetch_feed(url))
        for item in items:
            uid = hashlib.md5((item["title"] + item["link"] + thesis["id"]).encode()).hexdigest()
            if uid in seen:
                continue
            seen.add(uid)
            if not is_relevant(item, thesis):
                continue
            signal, reason = classify_with_llm(item["title"], item["desc"], thesis)
            if signal != "NONE":
                post_discord(signal, thesis["id"], thesis["tickers"], item["title"], item["link"], reason)
                alerts += 1
    return seen, alerts

def main():
    mode = "Haiku/OpenRouter" if OPENROUTER_KEY else "keyword fallback"
    log.info(f"News watcher starting | classifier: {mode} | discord: {'on' if DISCORD_WEBHOOK else 'off'}")
    seen = load_seen()
    while True:
        try:
            seen, n = run_once(seen)
            save_seen(seen)
            log.info(f"Cycle done — {n} alerts")
        except Exception as e:
            log.error(f"Cycle error: {e}")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
