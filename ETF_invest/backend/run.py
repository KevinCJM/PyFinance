from __future__ import annotations

import sys
import time
import uvicorn
import threading
import webbrowser
from pathlib import Path


def open_browser_when_ready(url: str, timeout: float = 8.0):
    start = time.time()
    # 简单延迟后直接尝试打开，避免依赖 requests
    time.sleep(1.0)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    # 兜底：超过超时不再处理
    while time.time() - start < timeout:
        time.sleep(0.2)


def main():
    port = 8000
    url = f"http://127.0.0.1:{port}"

    # 检查前端是否已构建
    dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
    if not dist.exists():
        print("[WARN] 未找到前端构建产物 frontend/dist。请先执行：\n  cd frontend && npm install && npm run build")

    t = threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True)
    t.start()

    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    sys.exit(main())
