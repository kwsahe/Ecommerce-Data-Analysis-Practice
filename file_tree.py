import os

def print_tree(root_path):
    for root, dirs, files in os.walk(root_path):
        level = root.replace(root_path, "").count(os.sep)
        indent = " " * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 4 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")

print_tree(".")