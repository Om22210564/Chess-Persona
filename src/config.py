from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

# Input PGN
PGN_FILE = RAW_DATA / "lichess_games.pgn"

# Your Lichess username
USERNAME = "Omkar22210564"