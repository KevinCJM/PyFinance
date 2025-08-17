# -*- encoding: utf-8 -*-
"""
@File: pdf_2_pic.py
@Modify Time: 2025/8/17 17:10       
@Author: Kevin-Chen
@Descriptions: 
"""
import fitz  # PyMuPDF
from PIL import Image
import os
import sys
import io


def get_target_size_in_bytes(size, unit):
    """根据输入的数值和单位，将其转换为字节。"""
    unit = unit.upper()
    if unit == 'KB':
        return size * 1024
    elif unit == 'MB':
        return size * 1024 * 1024
    else:
        raise ValueError(f"不支持的单位 '{unit}'。请使用 'KB' 或 'MB'。")


def compress_image_in_memory(pil_image, target_bytes):
    """
    在内存中对Pillow图片对象进行压缩，直到其大小接近目标。

    :param pil_image: Pillow 的 Image 对象。
    :param target_bytes: 目标文件大小（字节）。
    :return: 包含压缩后图片数据的 bytes 对象。
    """
    # 确保图片是 RGB 模式，因为 JPEG 不支持透明度
    if pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")

    # 使用二分法在内存中查找最佳质量
    low = 1
    high = 95
    best_quality = low

    while low <= high:
        quality = (low + high) // 2
        # 创建一个内存中的字节缓冲区
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)

        # 获取缓冲区中的数据大小
        current_size = buffer.tell()

        if current_size <= target_bytes:
            best_quality = quality
            low = quality + 1
        else:
            high = quality - 1

    # 使用找到的最佳质量，生成最终的图片数据
    final_buffer = io.BytesIO()
    pil_image.save(final_buffer, format='JPEG', quality=best_quality)

    final_size_kb = final_buffer.tell() / 1024
    print(f"    - 压缩质量: {best_quality}, 最终大小: {final_size_kb:.2f} KB")

    return final_buffer.getvalue()


def convert_and_compress_pdf(pdf_path, target_size, target_unit):
    """
    转换PDF为图片，并将每张图片压缩到指定大小。
    输出格式强制为 JPEG。
    """
    try:
        target_bytes = get_target_size_in_bytes(target_size, target_unit)
    except ValueError as e:
        print(f"错误: {e}")
        return

    if not os.path.exists(pdf_path):
        print(f"错误: 文件 '{pdf_path}' 不存在。")
        return

    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_format = 'jpeg'

    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)

        print(f"PDF共有 {num_pages} 页。目标大小: {target_size} {target_unit.upper()}")

        # 根据页数决定输出路径
        if num_pages == 1:
            page = doc.load_page(0)
            print(f"正在处理第 1 页...")

            # 1. 从PDF页面创建高分辨率Pillow图片
            # 先用较高的DPI渲染，以保证压缩源的质量
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes()))

            # 2. 在内存中压缩图片到目标大小
            compressed_data = compress_image_in_memory(img, target_bytes)

            # 3. 将压缩后的数据写入文件
            output_path = f"{base_name}.{output_format}"
            with open(output_path, "wb") as f:
                f.write(compressed_data)
            print(f"\n转换成功! 图片已保存到: {output_path}")

        else:
            output_folder = base_name
            os.makedirs(output_folder, exist_ok=True)
            print(f"正在创建并保存到文件夹: {output_folder}")

            for page_num in range(num_pages):
                page = doc.load_page(page_num)
                print(f"\n正在处理第 {page_num + 1} 页...")

                # 同样，先渲染再压缩
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes()))
                compressed_data = compress_image_in_memory(img, target_bytes)

                output_path = os.path.join(
                    output_folder,
                    f"{base_name}_page_{page_num + 1}.{output_format}"
                )
                with open(output_path, "wb") as f:
                    f.write(compressed_data)
                print(f"  -> 已保存页面 {page_num + 1} 到 {output_path}")

            print(f"\n所有页面转换完成! 文件保存在文件夹 '{output_folder}' 中。")

    except Exception as e:
        print(f"处理PDF时发生严重错误: {e}")
    finally:
        if 'doc' in locals() and doc:
            doc.close()


# --- 使用示例 ---
if __name__ == "__main__":
    pdf_file = "/Users/chenjunming/Desktop/个人信息/UNSW&UBC毕业证/UBC毕业证.pdf"
    print("\n--- PDF 图片转换与压缩工具 ---")
    print("注意: 输出格式将是JPEG以确保能控制文件大小。")

    try:
        target_input = input("请输入每张图片的目标大小 (例如: 500 KB, 2 MB): ")
        parts = target_input.strip().split()
        target_value = float(parts[0])
        target_unit_str = parts[1]
    except (IndexError, ValueError):
        print("输入格式错误。将使用默认值 '500 KB'。")
        target_value = 500
        target_unit_str = 'KB'

    print("-" * 30)

    # 调用主函数
    convert_and_compress_pdf(pdf_file, target_value, target_unit_str)
