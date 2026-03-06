from content_part1 import content_map as map1
from content_part2 import content_map as map2
from content_part3 import content_map as map3
from content_part4 import content_map as map4

from fill_engine import fill_notebook

def merge_maps(*maps):
    merged = {}
    for m in maps:
        merged.update(m)
    return merged

if __name__ == '__main__':
    final_map = merge_maps(map1, map2, map3, map4)
    fill_notebook('p:/Project/Labmentix/Deep_Cast/Sample_ML_Submission_Template-2.ipynb',
                  final_map,
                  'p:/Project/Labmentix/Deep_Cast/Sample_ML_Submission_Template-2.ipynb')
    print("Notebook perfectly injected and updated.")
