import yaml
from pathlib import Path

def get_config_data():
  """Loads configuration data from the YAML file."""
  config_file_path = Path(__file__).parent.parent / "configs" / "config.yaml"
  with config_file_path.open() as config_file:
    config_data = yaml.safe_load(config_file)
  return config_data