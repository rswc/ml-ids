from pathlib import Path

def get_project_root() -> Path:
    """ Return Path object representing absolute path to the project root """
    return Path(__file__).parent.parent