from maia_dataset import MaiaDataset

dataset = MaiaDataset("data/raw/sample.pgn")

print("Samples :", len(dataset))

x, policy, value = dataset[0]
# tokens, policy, value = dataset[0]

print(x.dtype)
# print(tokens.shape)
print("Input :", x.shape)
print("Policy :", policy.item())
print("Value :", value.item())