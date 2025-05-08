import subprocess
#
# 4. PDFCompressor（Python + Ghostscript）
# 许可证：MIT
#
# 特点：Python封装Ghostscript，提供更友好的API。
#
# 安装：
#
# bash
# pip install pdfcompressor
# 代码示例：
#
# python
# from pdfcompressor import compress
#
# compress("input.pdf", "output.pdf", power=3)  # power=1~4（1最佳质量，4最小体积）

# pdf 文件压缩，效果非常好，但需要安装 Ghostscript
# 在 Windows 上安装 Ghostscript 的步骤如下：
#
# 方法1：通过官方安装包（推荐）
# 下载安装包：
#
# 访问 Ghostscript 官网：https://www.ghostscript.com
def compress_pdf_ghostscript(input_path, output_path, quality=3):
    """
    quality参数（推荐用3或4）:
    0: 默认（不推荐）
    1: /prepress （高质量，文件大）
    2: /printer （高质量打印）
    3: /ebook （中等质量，适合电子书，推荐）
    4: /screen （低质量，文件最小）
    """
    quality_settings = {
        0: "/default",
        1: "/prepress",
        2: "/printer",
        3: "/ebook",
        4: "/screen"
    }

    command = [
        "gswin64c",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.4",
        f"-dPDFSETTINGS={quality_settings[quality]}",
        "-dNOPAUSE",
        "-dQUIET",
        "-dBATCH",
        f"-sOutputFile={output_path}",
        input_path
    ]

    subprocess.run(command, check=True)

# 使用示例（推荐quality=3）
compress_pdf_ghostscript("D:/油小酷/秒到/油小酷品牌连锁/附件5_式予以标注和呈现的条款(2).pdf", "D:/油小酷/秒到/油小酷品牌连锁/附件5_式予以标注和呈现的条款(2)-3.pdf",4)