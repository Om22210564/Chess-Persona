import hashlib
import json
import random
import re
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIA3_ROOT = PROJECT_ROOT / "external" / "maia3"
if MAIA3_ROOT.exists() and str(MAIA3_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIA3_ROOT))

import chess
import chess.pgn
import torch
from torch.utils.data import Dataset

from maia3.dataset import tokenize_board, get_historical_tokens
from maia3.utils import get_all_possible_moves, mirror_move


ALL_MOVES = get_all_possible_moves()
MOVE_TO_INDEX = {m: i for i, m in enumerate(ALL_MOVES)}
DEFAULT_ELO = 1500


def parse_elo(value, default=DEFAULT_ELO):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_time_control(time_control: Optional[str]):
    """Return (base_seconds, increment_seconds) from a PGN TimeControl tag."""
    if not time_control or time_control in {"-", "?"}:
        return 0, 0

    # Lichess commonly uses "300+3". Ignore complex multi-stage controls for now.
    match = re.match(r"^(\d+)(?:\+(\d+))?$", time_control)
    if not match:
        return 0, 0

    base = int(match.group(1))
    inc = int(match.group(2) or 0)
    return base, inc


def clock_to_seconds(clock_str: str):
    parts = clock_str.split(":")
    if len(parts) != 3:
        return None

    try:
        hours, minutes, seconds = map(int, parts)
    except ValueError:
        return None

    return hours * 3600 + minutes * 60 + seconds


def extract_clock(comment: str):
    if not comment:
        return None

    match = re.search(r"%clk\s+(\d+:\d+:\d+)", comment)
    if not match:
        return None

    return clock_to_seconds(match.group(1))


def result_value_for_side(result: str, side: bool):
    """Class label from the side-to-move perspective: loss=0, draw=1, win=2."""
    if result == "1/2-1/2" or result == "*":
        return 1
    if result == "1-0":
        return 2 if side == chess.WHITE else 0
    if result == "0-1":
        return 2 if side == chess.BLACK else 0
    return 1


class MaiaDataset(Dataset):
    def __init__(
        self,
        pgn_path: Union[str, Path],
        history: int = 8,
        username: Optional[str] = None,
        only_user_moves: bool = False,
        include_elos: bool = False,
        include_time_info: bool = False,
        default_elo: int = DEFAULT_ELO,
        split: str = "all",
        val_fraction: float = 0.0,
        split_seed: int = 42,
        cache_path: Optional[Union[str, Path]] = None,
        rebuild_cache: bool = False,
        log_stats: bool = False,
    ):
        self.history = history
        self.username = username
        self.only_user_moves = only_user_moves
        self.include_elos = include_elos
        self.include_time_info = include_time_info
        if split not in {"all", "train", "val"}:
            raise ValueError("split must be one of: all, train, val")
        if not 0 <= val_fraction < 1:
            raise ValueError("val_fraction must be >= 0 and < 1")
        if only_user_moves and not username:
            raise ValueError("username is required when only_user_moves=True")

        self.default_elo = default_elo
        self.split = split
        self.val_fraction = val_fraction
        self.split_seed = split_seed
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.rebuild_cache = rebuild_cache
        self.samples = []
        self.stats: Dict[str, int | float] = {
            "games": 0,
            "samples": 0,
            "skipped_non_user_moves": 0,
            "skipped_unknown_moves": 0,
            "white_games": 0,
            "black_games": 0,
            "unknown_user_games": 0,
            "skipped_split_games": 0,
            "rating_sum": 0,
            "rating_count": 0,
            "average_rating": 0.0,
        }

        self.cache_metadata = self._build_cache_metadata(pgn_path)

        if self.cache_path and not rebuild_cache and self._load_cache_if_valid():
            if log_stats:
                self.print_stats()
            return

        cfg = SimpleNamespace(
            history=history,
            include_time_info=include_time_info,
        )

        rng = random.Random(split_seed)

        with open(pgn_path, encoding="utf-8") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                is_val_game = val_fraction > 0 and rng.random() < val_fraction
                if split == "train" and is_val_game:
                    self.stats["skipped_split_games"] += 1
                    continue
                if split == "val" and not is_val_game:
                    self.stats["skipped_split_games"] += 1
                    continue

                self.stats["games"] += 1
                self._add_game_samples(game, cfg)

        self.stats["samples"] = len(self.samples)
        if self.stats["rating_count"]:
            self.stats["average_rating"] = round(
                self.stats["rating_sum"] / self.stats["rating_count"],
                2,
            )

        if self.cache_path:
            self._save_cache()

        if log_stats:
            self.print_stats()

    def _build_cache_metadata(self, pgn_path):
        pgn_path = Path(pgn_path)
        stat = pgn_path.stat()
        config = {
            "pgn_path": str(pgn_path.resolve()),
            "pgn_mtime_ns": stat.st_mtime_ns,
            "pgn_size": stat.st_size,
            "history": self.history,
            "username": self.username,
            "only_user_moves": self.only_user_moves,
            "include_elos": self.include_elos,
            "include_time_info": self.include_time_info,
            "default_elo": self.default_elo,
            "split": self.split,
            "val_fraction": self.val_fraction,
            "split_seed": self.split_seed,
        }
        config_json = json.dumps(config, sort_keys=True)
        return {
            "version": 1,
            "config": config,
            "config_hash": hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
        }

    def _load_cache_if_valid(self):
        cache_path = self.cache_path
        if cache_path is None or not cache_path.exists():
            return False

        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        if payload.get("metadata") != self.cache_metadata:
            return False

        self.samples = payload["samples"]
        self.stats = payload["stats"]
        return True

    def _save_cache(self):
        cache_path = self.cache_path
        if cache_path is None:
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": self.cache_metadata,
                "stats": self.stats,
                "samples": self.samples,
            },
            cache_path,
        )

    def _user_color(self, headers):
        if self.username is None:
            return None

        username = self.username.lower()
        if headers.get("White", "").lower() == username:
            return chess.WHITE
        if headers.get("Black", "").lower() == username:
            return chess.BLACK
        return None

    def _add_game_samples(self, game, cfg):
        headers = game.headers
        user_color = self._user_color(headers)

        if user_color == chess.WHITE:
            self.stats["white_games"] += 1
        elif user_color == chess.BLACK:
            self.stats["black_games"] += 1
        elif self.username is not None:
            self.stats["unknown_user_games"] += 1

        white_elo = parse_elo(headers.get("WhiteElo"), self.default_elo)
        black_elo = parse_elo(headers.get("BlackElo"), self.default_elo)
        base, inc = parse_time_control(headers.get("TimeControl"))
        clocks = {chess.WHITE: base, chess.BLACK: base}

        board = game.board()
        history_queue = deque(maxlen=self.history)
        history_queue.append(tokenize_board(board))

        node = game
        while node.variations:
            next_node = node.variation(0)
            move = next_node.move
            side_to_move = board.turn

            if self.only_user_moves and side_to_move != user_color:
                self.stats["skipped_non_user_moves"] += 1
                board.push(move)
                after_clock = extract_clock(next_node.comment)
                if after_clock is not None:
                    clocks[side_to_move] = after_clock
                history_queue.append(tokenize_board(board))
                node = next_node
                continue

            move_uci = move.uci()
            if side_to_move == chess.BLACK:
                move_uci = mirror_move(move_uci)

            if move_uci not in MOVE_TO_INDEX:
                self.stats["skipped_unknown_moves"] += 1
                board.push(move)
                after_clock = extract_clock(next_node.comment)
                if after_clock is not None:
                    clocks[side_to_move] = after_clock
                history_queue.append(tokenize_board(board))
                node = next_node
                continue

            clk_left_before = clocks.get(side_to_move, 0) or 0
            self_elo = white_elo if side_to_move == chess.WHITE else black_elo
            oppo_elo = black_elo if side_to_move == chess.WHITE else white_elo
            self.stats["rating_sum"] += self_elo
            self.stats["rating_count"] += 1
            value = result_value_for_side(headers.get("Result", "*"), side_to_move)

            tokens = get_historical_tokens(
                history_queue,
                cfg,
                base=base,
                inc=inc,
                clk_left_before=clk_left_before,
                clk_ponder=0,
            )

            if self.include_elos:
                self.samples.append(
                    (tokens, MOVE_TO_INDEX[move_uci], value, self_elo, oppo_elo)
                )
            else:
                self.samples.append((tokens, MOVE_TO_INDEX[move_uci], value))

            board.push(move)
            after_clock = extract_clock(next_node.comment)
            if after_clock is not None:
                # Comments are attached to the position after the move; the moving side's
                # remaining clock is therefore available after board.push().
                clocks[side_to_move] = after_clock
            history_queue.append(tokenize_board(board))
            node = next_node

    def print_stats(self):
        print("MaiaDataset statistics:")
        for key, value in self.stats.items():
            print(f"  {key}: {value}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.include_elos:
            x, policy, value, self_elo, oppo_elo = sample
            return (
                x.float(),
                torch.tensor(policy, dtype=torch.long),
                torch.tensor(value, dtype=torch.long),
                torch.tensor(self_elo, dtype=torch.long),
                torch.tensor(oppo_elo, dtype=torch.long),
            )

        x, policy, value = sample
        return (
            x.float(),
            torch.tensor(policy, dtype=torch.long),
            torch.tensor(value, dtype=torch.long),
        )
