import zipfile
import xml.etree.ElementTree as ET
import sys

def main():
    try:
        z = zipfile.ZipFile('p:/Project/Labmentix/Deep_Cast/DeepCSAT – Ecommerce.pptx')
        slides = [f for f in z.namelist() if f.startswith('ppt/slides/slide')]
        for f in sorted(slides):
            print(f"--- {f} ---")
            xml_content = z.read(f)
            tree = ET.fromstring(xml_content)
            for node in tree.iter():
                if node.tag.endswith('}t') and node.text:
                    print(node.text)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
