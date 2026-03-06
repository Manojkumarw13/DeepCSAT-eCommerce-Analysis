import json
import sys

def main():
    try:
        with open('p:/Project/Labmentix/Deep_Cast/Sample_ML_Submission_Template-2.ipynb', 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        with open('p:/Project/Labmentix/Deep_Cast/nb_structure.txt', 'w', encoding='utf-8') as out:
            for i, cell in enumerate(nb['cells']):
                cell_type = cell['cell_type']
                source = ''.join(cell.get('source', []))
                snippet = source[:100].replace('\n', ' ')
                out.write(f"Cell {i} [{cell_type}]: {snippet}\n")
        print("Structure saved to nb_structure.txt")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
