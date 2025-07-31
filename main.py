import load_data

data_loader = load_data.Dataloader("data_ikea")

data_struct = data_loader.structure

for cls in data_struct:
    print(f"{cls}: {len(data_struct[cls])} items")


