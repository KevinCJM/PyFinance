import time
import os
import sys
import socket
import click

# 将当前运行路径添加进系统环境变量
run_path = os.getcwd()
sys.path.extend([run_path])

from server.base import app


def get_local_ip():
    """获取本机局域网 IP 地址"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 这里不需要真实连通，只是为了获取本机的出口 IP
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def run_server():
    # 这里应该配置一些选项，例如是否缓存、是否初始化股票收益等
    local_ip = get_local_ip()
    print("本机局域网 IP 地址: http://%s:6001" % local_ip)
    print("Flask 路由映射:\n", app.url_map)
    app.run(debug=False, host="0.0.0.0", port=6001)


def run():
    run_server()


if __name__ == "__main__":
    start_t = time.time()
    run()
    print("use time: %s" % (time.time() - start_t))
