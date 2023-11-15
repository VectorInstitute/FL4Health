# import pickle

# id = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

# paths = [f'examples/secure_aggregation_example/{i}.pkl' for i in id]

# for path in paths:
#    a = pickle.load(open(path, "rb"))
#    for k, v in a.items():
#       print(v.dtype, v.shape)

with open(f"examples/secure_aggregation_example/a.txt", "w") as f:
    f.write("123")
