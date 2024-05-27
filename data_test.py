import pickle

with open("SIMS.pkl", 'rb') as file:
    data = pickle.load(file)
print(data.keys())
