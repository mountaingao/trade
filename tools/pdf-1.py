import pypdfium2 as pdfium

# pip install PyPDF2 pdf2image img2pdf pypdfium2
# pip install  pypdfium2
def compress_pdf_pypdfium2(input_path, output_path):
    pdf = pdfium.PdfDocument(input_path)

    # 设置压缩选项
    options = pdfium.PdfSaveOptions(
        compress=True,
        compress_images=True,
        no_metadata=True,
    )

    pdf.save(output_path, options)

# 使用示例
compress_pdf_pypdfium2("D:/油小酷/秒到/油小酷品牌连锁/附件5_式予以标注和呈现的条款(2).pdf", "D:/油小酷/秒到/油小酷品牌连锁/附件5_式予以标注和呈现的条款(2)-2.pdf")