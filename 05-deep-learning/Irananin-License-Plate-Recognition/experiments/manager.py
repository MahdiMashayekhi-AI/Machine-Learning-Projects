import os
import re

def create_experiment(root_dir="experiments"):
    os.makedirs(root_dir, exist_ok=True)

    exp_ids = []
    for d in os.listdir(root_dir):
      if re.fullmatch(r"exp_\d{3}", d):
        exp_ids.append(int(d.split("_")[1]))

    next_id = max(exp_ids) + 1 if exp_ids else 1
    exp_name = f"exp_{next_id:03d}"
    exp_path = os.path.join(root_dir, exp_name)

    os.makedirs(os.path.join(exp_path, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_path, "logs"), exist_ok=True)
  
    return exp_path