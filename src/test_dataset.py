from dataset import PGNParser
import numpy as np

parser = PGNParser(
    "data/raw/sample.pgn",
    history_length=8,
)

samples = parser.parse_games(max_games=1)

history, policy, value = samples[0]

history = np.stack(history)

print("History shape :", history.shape)
print("Policy :", policy)
print("Value :", value)