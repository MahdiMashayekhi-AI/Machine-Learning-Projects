import os
from pathlib import Path

ALLOWED_EXTENSIONS = {
    ".py", ".yaml", ".yml", ".json", ".md"
}

IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".idea",
    ".vscode",
    "node_modules"
}

def print_tree(start_path):
    start_path = Path(start_path)

    for root, dirs, files in os.walk(start_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        level = Path(root).relative_to(start_path).parts
        indent = "│   " * len(level)
        folder_name = Path(root).name if level else start_path.name

        print(f"{indent}📂 {folder_name}")

        sub_indent = "│   " * (len(level) + 1)

        for file in files:
            ext = Path(file).suffix.lower()
            if ext in ALLOWED_EXTENSIONS:
                print(f"{sub_indent}📄 {file}")


if __name__ == "__main__":
    print_tree(".")
