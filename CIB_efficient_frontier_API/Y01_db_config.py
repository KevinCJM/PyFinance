# -*- encoding: utf-8 -*-
"""
@File: Y01_db_config.py.py
@Modify Time: 2025/9/20 17:31
@Author: Kevin-Chen
@Descriptions: 数据库配置信息与连通性测试（自动识别容器/本机）
"""

import os
import socket
from typing import Optional
from sqlalchemy import create_engine, text

# 基础账号库名配置（敏感信息按需替换）
db_type = 'mysql'
db_port = '3306'
db_name = 'mysql'
db_user = 'root'
db_password = '112358'


def _is_in_docker() -> bool:
    """粗略判断当前是否运行于容器中。"""
    if os.path.exists('/.dockerenv'):
        return True
    try:
        with open('/proc/1/cgroup', 'rt') as f:
            data = f.read()
            if 'docker' in data or 'kubepods' in data or 'containerd' in data:
                return True
    except Exception:
        pass
    return os.environ.get('RUNNING_IN_DOCKER', '').lower() in ('1', 'true', 'yes')


def _resolve_host_for_container() -> str:
    """在容器内解析宿主机地址。

    优先使用 host.docker.internal；解析失败则回退到常见网桥网关 172.17.0.1，
    也可通过环境变量 HOST_DOCKER_GATEWAY 覆盖。
    """
    host = 'host.docker.internal'
    try:
        socket.gethostbyname(host)
        return host
    except Exception:
        return os.environ.get('HOST_DOCKER_GATEWAY', '172.17.0.1')


def _default_host() -> str:
    return _resolve_host_for_container() if _is_in_docker() else '127.0.0.1'


def _build_db_url(host: Optional[str] = None) -> str:
    host = host or _default_host()
    if db_type == 'mysql':
        db_driver = 'pymysql'
        return f'mysql+{db_driver}://{db_user}:{db_password}@{host}:{db_port}/{db_name}?charset=utf8mb4'
    raise ValueError('Unsupported db_type: {}'.format(db_type))


# 兼容旧代码：导出一个默认 db_url（基于自动 host 判定）
db_url = _build_db_url()


def get_active_db_url() -> str:
    """返回最终使用的数据库连接串。优先使用环境变量 DB_URL。"""
    return os.environ.get('DB_URL') or db_url


def try_connect(url: Optional[str] = None, timeout: int = 5) -> bool:
    """尝试连接数据库并执行简单探活查询。"""
    url = url or get_active_db_url()
    try:
        engine = create_engine(url, connect_args={'connect_timeout': timeout}, pool_pre_ping=True)
        with engine.connect() as conn:
            ver = conn.execute(text('SELECT VERSION()')).scalar()
            one = conn.execute(text('SELECT 1')).scalar()
        print('[OK] 数据库连接成功 | VERSION={} | SELECT 1 -> {}'.format(ver, one))
        return True
    except Exception as e:
        print('[ERROR] 数据库连接失败: {}\nURL={}'.format(e, url))
        return False


if __name__ == '__main__':
    src = 'ENV(DB_URL)' if os.environ.get('DB_URL') else (
        'Container(auto-host)' if _is_in_docker() else 'Local(127.0.0.1)')
    active = get_active_db_url()
    print(f'[INFO] 使用连接串来源: {src}')
    ok = try_connect(active)
    exit(0 if ok else 2)
