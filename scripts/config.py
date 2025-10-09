from __future__ import annotations
import os, time, pathlib

# --- Monte Carlo precision ---
MONTE_CARLO_TRIALS = int(os.getenv("MC_TRIALS", "25000"))  # default 25k

# --- Logging ---
RUN_TS  = os.getenv("RUN_TS", str(int(time.time())))
RUN_ID  = os.getenv("RUN_ID", RUN_TS)                 # you can pass a custom id in Actions
LOG_DIR = pathlib.Path(os.getenv("LOG_DIR", "logs/daily_runs")) / RUN_ID
LOG_DIR.mkdir(parents=True, exist_ok=True)
