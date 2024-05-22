import os
import pickle
import yaml
from pathlib import Path


def get_config_data():
  """Loads configuration data from the YAML file."""
  config_file_path = Path(__file__).parent.parent / "configs" / "config.yaml"
  with config_file_path.open() as config_file:
    config_data = yaml.safe_load(config_file)
  return config_data


def save_object_pickle(file_path: str, obj: object) -> None:
    """Saves an object using pickle.

    Args:
        file_path (str): Path to the file where the object will be saved.
        obj (object): The object to be saved.

    Raises:
        CustomException: If an error occurs during saving.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise e
    
def load_object(file_path: str) -> object:
    """Loads an object previously saved using pickle.

    Args:
        file_path (str): Path to the file containing the pickled object.

    Returns:
        object: The loaded object.

    Raises:
        CustomException: If an error occurs during loading.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise e