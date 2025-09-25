# -*- encoding: utf-8 -*-
"""
@File: Y01_db_config.py.py
@Modify Time: 2025/9/20 17:31
@Author: Kevin-Chen
@Descriptions: 数据库配置信息（仅配置）；主程序内使用 T05_db_utils 进行连通性测试
"""

# 数据库配置信息（如需覆盖主机/IP，可设置 db_host；None 表示自动判定）
db_type = 'mysql'
db_host = None  # None -> 自动：容器内使用 host.docker.internal/网关，本机使用 127.0.0.1
db_port = '3306'
db_name = 'mysql'
db_user = 'root'
db_password = '112358'

if __name__ == '__main__':
    from T05_db_utils import get_active_db_url, try_connect, is_in_docker
    import os as _os

    url = get_active_db_url(
        db_type=db_type,
        db_user=db_user,
        db_password=db_password,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
    )
    src = 'ENV(DB_URL)' if 'DB_URL' in _os.environ else (
        'Container(auto-host)' if is_in_docker() else 'Local(127.0.0.1)'
    )
    print(f'[INFO] 使用连接串来源: {src}')
    ok = try_connect(url)
    if ok:
        print(f'[OK] 成功连接到数据库')
    exit(0 if ok else 2)
