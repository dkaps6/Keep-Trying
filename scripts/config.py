from __future__ import annotations
import os, time, pathlib
MONTE_CARLO_TRIALS=int(os.getenv('MC_TRIALS','25000'))
RUN_TS=os.getenv('RUN_TS',str(int(time.time())))
RUN_ID=os.getenv('RUN_ID',RUN_TS)
LOG_DIR=pathlib.Path(os.getenv('LOG_DIR','logs/daily_runs'))/RUN_ID
LOG_DIR.mkdir(parents=True, exist_ok=True)
