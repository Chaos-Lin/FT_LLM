import pickle

with open("MOSEI.pkl", 'rb') as file:
    data = pickle.load(file)
print(data.keys())
print(data["train"]["audio"].shape)
