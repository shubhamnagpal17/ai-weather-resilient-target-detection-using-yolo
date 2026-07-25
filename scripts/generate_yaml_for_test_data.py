def make_yaml(variant_name):
    content = f"""
path: datasets/BDD100K
train: splits/train.txt
val: splits/val.txt
test: images/{variant_name}

names:
  0: person
  1: rider
  2: car
  3: truck
  4: train
  5: bike
"""
    yaml_path = f"configs/data_{variant_name}.yaml"
    with open(yaml_path, "w") as f:
        f.write(content)
    return yaml_path

variant_names = ["test_clahe", "test_gamma", "test_denoise", "test_full_pipeline"]
variant_yamls = {name: make_yaml(name) for name in variant_names}