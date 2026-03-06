import json
import shutil

def fill_notebook(notebook_path, content_map, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for idx, content in content_map.items():
        if idx < len(nb['cells']):
            # Convert string to list of distinct lines with newline chars to maintain valid jupyter syntax
            lines = [line + '\n' for line in content.split('\n')]
            # Remove the trailing newline from the last line
            if lines:
                lines[-1] = lines[-1].rstrip('\n')
            nb['cells'][idx]['source'] = lines
            
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Filled notebook saved to {output_path}")

# We will populate content_map in another file and import this.
