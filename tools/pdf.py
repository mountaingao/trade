import fitz  # 导入fitz模块


import fitz

def compress_pdf_1(input_path, output_path):
    doc = fitz.open(input_path)
    doc.save(output_path, garbage=4, clean=False)
    doc.close()


# pip install pymupdf
def compress_pdf(input_path, output_path, resolution=100):
    # 打开PDF文件
    doc = fitz.open(input_path)

    # 创建一个新的 PDF 文件
    compressed_doc = fitz.open()

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # 压缩页面图像
        pix = page.get_pixmap(matrix=fitz.Matrix(resolution / 100, resolution / 100))
        img = fitz.Pixmap(pix)

        # 创建新的页面并插入压缩后的图像
        new_page = compressed_doc.new_page(width=img.width, height=img.height)
        new_page.insert_image(new_page.rect, pixmap=img)

    # 保存压缩后的 PDF 文件
    compressed_doc.save(output_path)
    compressed_doc.close()
    doc.close()

# 使用函数
input_pdf = 'D:/油小酷/秒到/油小酷品牌连锁/附件5_式予以标注和呈现的条款(2).pdf'
output_pdf = 'D:/油小酷/秒到/油小酷品牌连锁/附件5_式予以标注和呈现的条款(2)-4.pdf'
# compress_pdf(input_pdf, output_pdf,50)
compress_pdf_1(input_pdf, output_pdf)

