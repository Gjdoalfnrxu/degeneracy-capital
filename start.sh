#!/bin/bash
# Start news watcher in background if OPENROUTER_API_KEY is set
if [ -n "$OPENROUTER_API_KEY" ]; then
  python newswatcher.py &
fi
exec uvicorn app:app --host 0.0.0.0 --port 8420
