import xml.etree.ElementTree as ET


def parse_xml_content(xml_string):
    root = ET.fromstring(xml_string)
    content = []
    for element in root.iter():
        if element.tag in {"description", "transcript"}:
            if element.text:
                content.append(element.text.strip())
    return content
