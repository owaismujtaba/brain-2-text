import os

def get_all_files(parent_dir, extensions=None):
    """
    Recursively loads all files in the parent directory and its subdirectories.
    If extensions is provided, only files with those extensions are returned.

    Args:
        parent_dir (str): Path to the parent directory.
        extensions (tuple, optional): File extensions to filter by, e.g. ('.h5', '.txt')

    Returns:
        List[str]: List of file paths.
    """
    all_files = []
    for root, _, files in os.walk(parent_dir):
        for file in files:
            if extensions is None or file.lower().endswith(extensions):
                all_files.append(os.path.join(root, file))
    return all_files