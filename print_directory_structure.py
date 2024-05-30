import os


def print_directory_structure(directory):
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        file_count = 0
        for file in files:
            if file_count < 3:
                print(f"{sub_indent}{file}")
                file_count += 1
            else:
                print(f"{sub_indent}...")
                break


directory = 'NEU-DET/labels'
print_directory_structure(directory)
