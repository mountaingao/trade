import xml.etree.ElementTree as ET

def add_stock_to_xml(stock_code, stock_name=""):
    file_path = r"D:\THS\userdata\自选股板块\自选股.xml"
    tree = ET.parse(file_path)
    root = tree.getroot()

    # 创建新股票节点
    new_stock = ET.SubElement(root, "Stock")
    new_stock.set("code", stock_code)
    new_stock.set("name", stock_name)

    # 保存修改
    tree.write(file_path, encoding="gbk", xml_declaration=True)

# 示例：添加宁德时代（300750）
add_stock_to_xml("300750", "宁德时代")