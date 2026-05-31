import pickle

def save_object(file_path, obj):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

def load_object(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
