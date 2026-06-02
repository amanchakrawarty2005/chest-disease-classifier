from pathlib import Path
root = Path('d:/chest-disease-classifier')
exclude_dirs = {'venv', '.git', '__pycache__'}
max_depth = 4

def tree(path, prefix='', depth=0):
    if depth > max_depth:
        return
    entries = sorted([p for p in path.iterdir() if p.name not in exclude_dirs], key=lambda x: (x.is_file(), x.name.lower()))
    for i, entry in enumerate(entries):
        connector = '`-- ' if i == len(entries) - 1 else '|-- '
        print(prefix + connector + entry.name)
        if entry.is_dir():
            extension = '    ' if i == len(entries) - 1 else '|   '
            tree(entry, prefix + extension, depth + 1)

print(root.name)
tree(root)
