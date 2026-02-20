import os
import yaml

def get_config():
  config_path = os.path.join(os.path.dirname(__file__), "settings.yaml")
  with open(config_path, "r") as f:
    config = yaml.safe_load(f)
  return config

cfg = get_config()