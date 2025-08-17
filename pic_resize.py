# -*- encoding: utf-8 -*-
"""
@File: pic_resize.py
@Modify Time: 2025/8/17 16:58       
@Author: Kevin-Chen
@Descriptions: 
"""
import os
from PIL import Image


def get_target_size_in_bytes(size, unit):
    """
    根据输入的数值和单位，将其转换为字节。

    :param size: 大小数值 (e.g., 2)
    :param unit: 单位字符串 (e.g., 'KB', 'MB', 'GB')
    :return: 对应的字节数
    """
    unit = unit.upper()
    if unit == 'KB':
        return size * 1024
    elif unit == 'MB':
        return size * 1024 * 1024
    elif unit == 'GB':
        return size * 1024 * 1024 * 1024
    else:
        # 如果单位不是 KB, MB, GB, 默认当作字节处理
        return size


def compress_image_to_target_size(input_path, output_path, target_size, target_unit):
    """
    将图片压缩到指定的大小和单位。

    :param input_path: 输入图片的路径
    :param output_path: 输出图片的路径
    :param target_size: 目标大小的数值
    :param target_unit: 目标大小的单位 ('KB', 'MB', etc.)
    """
    target_size_bytes = get_target_size_in_bytes(target_size, target_unit)
    if not target_size_bytes:
        print(f"错误: 不支持的单位 '{target_unit}'。请使用 'KB', 'MB', 或 'GB'。")
        return

    try:
        # 打开图片
        with Image.open(input_path) as img:
            # 获取原始格式
            original_format = img.format
            # 如果要进行有效的有损压缩以控制大小，最好转换为JPEG
            # 这里我们假设目标是减小文件大小，所以统一输出为JPEG
            # 如果需要保留PNG透明度等特性，则逻辑会更复杂
            output_format = 'JPEG'

            print(f"原始格式: {original_format}。为有效压缩，将统一保存为 JPEG 格式。")

            # 使用二分法来寻找最佳质量参数
            low = 1
            high = 95
            best_quality = low

            # 为了处理RGBA (带透明通道) 的图片，如PNG，需要先转换为RGB
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # 迭代寻找最接近目标大小的质量设置
            # 为了避免无限循环，设置一个迭代次数限制
            for _ in range(10):  # 10次迭代对于二分法已经足够
                quality = (low + high) // 2
                if quality == 0:  # 避免质量为0
                    break

                temp_output_path = "temp_image.jpg"
                img.save(temp_output_path, format=output_format, quality=quality, optimize=True)

                current_size_bytes = os.path.getsize(temp_output_path)

                # 如果当前大小已经小于目标，可以尝试提高质量
                if current_size_bytes <= target_size_bytes:
                    best_quality = quality
                    low = quality + 1
                else:  # 如果当前大小大于目标，需要降低质量
                    high = quality - 1

            # 使用找到的最佳质量参数保存最终图片
            img.save(output_path, format=output_format, quality=best_quality, optimize=True)
            if os.path.exists("temp_image.jpg"):
                os.remove("temp_image.jpg")

            final_size_bytes = os.path.getsize(output_path)
            final_size_kb = final_size_bytes / 1024
            final_size_mb = final_size_kb / 1024

            print("-" * 30)
            print(f"压缩完成!")
            print(f"图片已保存到: {output_path}")
            print(f"最佳压缩质量: {best_quality}")
            if final_size_mb >= 1:
                print(f"最终文件大小: {final_size_mb:.2f} MB")
            else:
                print(f"最终文件大小: {final_size_kb:.2f} KB")

            if final_size_bytes > target_size_bytes:
                print(f"警告: 最终大小略高于目标。这可能是因为在质量为 {best_quality} 时文件大小已是最低的可达值。")


    except FileNotFoundError:
        print(f"错误: 输入文件 '{input_path}' 不存在。")
    except Exception as e:
        print(f"处理图片时发生错误: {e}")


# --- 使用示例 ---
if __name__ == "__main__":
    # --- 1. 设置参数 ---
    input_image_path = "/Users/chenjunming/Desktop/我的/招商银行卡反面.png"  # <-- 修改为你的图片路径
    output_image_path = "compressed_image.jpg"  # <-- 输出文件名，格式将是JPEG

    # --- 2. 从用户获取目标大小 ---
    try:
        target_input = input("请输入目标文件大小 (例如: 2 MB, 500 KB): ")
        # 分割输入字符串获取数值和单位
        parts = target_input.strip().split()
        target_value = float(parts[0])
        target_unit_str = parts[1]
    except (IndexError, ValueError):
        print("输入格式错误。将使用默认值 '2 MB'。")
        target_value = 2.0
        target_unit_str = 'MB'

    print("-" * 30)
    print(f"开始压缩图片: '{input_image_path}'")
    print(f"目标大小: {target_value} {target_unit_str}")

    # --- 3. 调用压缩函数 ---
    compress_image_to_target_size(input_image_path, output_image_path, target_value, target_unit_str)
