import os

def _create_config_manager(): 
    """This function is implemented so that the root_dir variable is only accessible through the nested set_root_dir and get_root_dir functions."""
    root_dir = os.getcwd()  # This variable is local to the closure

    def set_root_dir(new_root_dir: str):
        nonlocal root_dir
        new_root_dir = os.path.expanduser(new_root_dir)
        if not os.path.exists(new_root_dir):
            os.makedirs(new_root_dir)
        root_dir = new_root_dir

    def get_root_dir():
        return root_dir

    return set_root_dir, get_root_dir

# Create the manager functions and only expose these.
set_root_dir, get_root_dir = _create_config_manager()

# Specify the public API.
__all__ = ['set_root_dir', 'get_root_dir']
