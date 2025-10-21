"""通用数据库连接与数据处理工具。

该模块基于 `SQLAlchemy <https://www.sqlalchemy.org/>`_ 提供数据库访问能力，
兼容 Python 3.6.9，并支持多种关系型数据库（MySQL、PostgreSQL、SQLite 等）。
核心功能包括：

* 构建数据库连接池并通过 ``with`` 语法安全获取连接
* 直接创建临时连接（无需显式管理连接池）
* 将查询结果读取为 :class:`pandas.DataFrame`
* 将 :class:`pandas.DataFrame` 以批量方式写入数据库
* 基于主键列批量更新数据库中的已有记录
* 提供结合线程池与连接池的批量读写工具函数

示例::

    from T07_db_utils import DatabaseConnectionPool, read_dataframe, insert_dataframe

    pool = DatabaseConnectionPool(
        url="mysql+pymysql://readonly:password@127.0.0.1:3306/sample_db",
        pool_size=4,
    )

    df = read_dataframe(pool, "SELECT * FROM sample_table WHERE biz_date = :biz_date", params={"biz_date": "2024-01-01"})

    insert_dataframe(pool, df, "sample_table_backup", replace=False)

    update_dataframe(pool, df, "sample_table", key_columns=["id"])

注意事项：
    * 需要安装 ``sqlalchemy``、``pandas`` 以及目标数据库对应的驱动（如 ``pymysql``、``psycopg2``、``cx_Oracle`` 等）。
    * ``url`` 参数遵循 SQLAlchemy 的连接字符串格式。
    * 模块不会自动创建数据库或数据表结构。
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

LOGGER = logging.getLogger(__name__)


def create_connection(url: str, **engine_kwargs: Any) -> Connection:
    """创建一个临时数据库连接。

    :param url: SQLAlchemy 连接字符串，例如 ``mysql+pymysql://user:pwd@host:3306/db``。
    :param engine_kwargs: 传递给 :func:`sqlalchemy.create_engine` 的其他参数。
    :return: 打开的 SQLAlchemy :class:`Connection` 对象，调用方负责关闭。
    """

    LOGGER.debug("创建独立数据库连接: url=%s", url)
    engine = create_engine(url, **engine_kwargs)
    return engine.connect()


class DatabaseConnectionPool(object):
    """基于 SQLAlchemy 的轻量级连接池封装。"""

    def __init__(
        self,
        url: str,
        pool_size: int = 4,
        max_overflow: Optional[int] = None,
        **engine_kwargs: Any
    ) -> None:
        if pool_size <= 0:
            raise ValueError("pool_size must be positive")

        if max_overflow is None:
            max_overflow = pool_size

        engine_kwargs.setdefault("pool_pre_ping", True)

        LOGGER.debug(
            "初始化连接池: url=%s, pool_size=%s, max_overflow=%s, kwargs=%s",
            url,
            pool_size,
            max_overflow,
            engine_kwargs,
        )

        self._engine = create_engine(
            url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            **engine_kwargs
        )

    @property
    def engine(self) -> Engine:
        """返回底层 SQLAlchemy :class:`Engine`。"""

        return self._engine

    def get_connection(self) -> Connection:
        """从连接池获取一个连接。"""

        return self._engine.connect()

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        """提供 ``with`` 语法获取连接。"""

        with self.get_connection() as conn:
            yield conn

    @contextmanager
    def begin(self) -> Iterator[Connection]:
        """开启一个自动提交的事务上下文。"""

        connection = self.get_connection()
        transaction = connection.begin()
        try:
            yield connection
        except Exception:
            transaction.rollback()
            raise
        else:
            transaction.commit()
        finally:
            connection.close()

    def dispose(self) -> None:
        """释放连接池资源。"""

        self._engine.dispose()


def read_dataframe(
    pool: DatabaseConnectionPool,
    query: str,
    params: Optional[Mapping[str, Any]] = None,
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    """执行查询并返回 :class:`pandas.DataFrame`。

    :param pool: 连接池实例。
    :param query: SQL 查询语句，可包含 SQLAlchemy ``:name`` 风格占位符。
    :param params: 查询参数，可为 ``dict`` 或 ``None``。
    :param chunksize: 指定时按块读取，返回迭代器。
    """

    LOGGER.debug("执行查询: %s, params=%s", query, params)
    return pd.read_sql_query(query, pool.engine, params=params, chunksize=chunksize)


def insert_dataframe(
    pool: DatabaseConnectionPool,
    dataframe: pd.DataFrame,
    table: str,
    schema: Optional[str] = None,
    replace: bool = False,
    batch_size: int = 1000,
    dtype: Optional[Dict[str, Any]] = None,
    method: Optional[str] = "multi",
) -> None:
    """将 :class:`pandas.DataFrame` 批量写入数据库表。

    :param pool: 连接池实例。
    :param dataframe: 待写入的数据。
    :param table: 目标表名。
    :param schema: 可选的 schema 名称。
    :param replace: ``True`` 时使用 ``if_exists='replace'``，否则 ``append``。
    :param batch_size: 每批写入的行数，对 ``to_sql`` 的 ``chunksize`` 参数。
    :param dtype: ``pandas.DataFrame.to_sql`` 的 ``dtype`` 映射，可用于指定列类型。
    :param method: ``to_sql`` 的 ``method`` 参数，默认 ``"multi"`` 以提升批量写入效率。
    """

    if dataframe.empty:
        LOGGER.info("DataFrame 为空，跳过写入: table=%s", table)
        return

    if_exists = "replace" if replace else "append"

    LOGGER.debug(
        "写入 DataFrame: table=%s, schema=%s, rows=%s, if_exists=%s",
        table,
        schema,
        len(dataframe),
        if_exists,
    )

    dataframe.to_sql(
        name=table,
        con=pool.engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        chunksize=batch_size,
        dtype=dtype,
        method=method,
    )


def update_dataframe(
    pool: DatabaseConnectionPool,
    dataframe: pd.DataFrame,
    table: str,
    key_columns: Sequence[str],
    schema: Optional[str] = None,
    batch_size: int = 500,
) -> None:
    """根据主键列批量更新数据库中的记录。

    :param pool: 连接池实例。
    :param dataframe: 含有更新数据的 DataFrame。
    :param table: 目标表名。
    :param key_columns: 主键列名集合，用于构造 ``WHERE`` 子句。
    :param schema: 可选的 schema 名称。
    :param batch_size: 每批次提交的行数。
    """

    if dataframe.empty:
        LOGGER.info("DataFrame 为空，跳过更新: table=%s", table)
        return

    for col in key_columns:
        if col not in dataframe.columns:
            raise ValueError("缺少主键列: {}".format(col))

    non_key_columns = [col for col in dataframe.columns if col not in key_columns]
    if not non_key_columns:
        raise ValueError("没有可更新的非主键列")

    dialect = pool.engine.dialect
    quote = dialect.identifier_preparer.quote

    def _quote_table(name: str) -> str:
        if schema:
            return ".".join(quote(part) for part in (schema, name))
        return ".".join(quote(part) for part in name.split("."))

    set_mapping = [(col, "set_{}".format(idx)) for idx, col in enumerate(non_key_columns)]
    where_mapping = [(col, "where_{}".format(idx)) for idx, col in enumerate(key_columns)]

    set_clause = ", ".join("{} = :{}".format(quote(col), param) for col, param in set_mapping)
    where_clause = " AND ".join("{} = :{}".format(quote(col), param) for col, param in where_mapping)

    sql = "UPDATE {} SET {} WHERE {}".format(_quote_table(table), set_clause, where_clause)
    statement = text(sql)

    LOGGER.debug(
        "批量更新数据: table=%s, schema=%s, rows=%s, key_columns=%s",
        table,
        schema,
        len(dataframe),
        key_columns,
    )

    try:
        connection = pool.get_connection()
        transaction = connection.begin()
        try:
            batch: list = []
            count = 0
            for row in dataframe.itertuples(index=False, name=None):
                row_dict = dict(zip(dataframe.columns, row))
                params: Dict[str, Any] = {}
                for col, param in set_mapping:
                    params[param] = row_dict[col]
                for col, param in where_mapping:
                    params[param] = row_dict[col]
                batch.append(params)
                if len(batch) >= batch_size:
                    connection.execute(statement, batch)
                    count += len(batch)
                    batch = []
            if batch:
                connection.execute(statement, batch)
                count += len(batch)
            transaction.commit()
            LOGGER.debug("更新完成，共影响行数: %s", count)
        except Exception:
            transaction.rollback()
            raise
        finally:
            connection.close()
    except SQLAlchemyError:
        LOGGER.exception("更新数据失败: table=%s", table)
        raise


def _run_with_thread_pool(
    worker: Callable[..., Any],
    tasks: Sequence[Tuple[int, Tuple[Any, ...], Dict[str, Any]]],
    max_workers: Optional[int] = None,
) -> Sequence[Any]:
    """使用线程池按照索引顺序执行任务。"""

    results: Dict[int, Any] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(worker, *args, **kwargs): index
            for index, args, kwargs in tasks
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()
    return [results[idx] for idx in sorted(results)]


def threaded_read_dataframe(
    pool: DatabaseConnectionPool,
    query_params: Sequence[Tuple[str, Optional[Mapping[str, Any]]]],
    max_workers: Optional[int] = None,
) -> Sequence[pd.DataFrame]:
    """使用线程池执行多条查询并返回多个 :class:`pandas.DataFrame`。

    :param pool: 连接池实例。
    :param query_params: ``(sql, params)`` 元组序列，``params`` 可为 ``None``。
    :param max_workers: 线程池最大线程数，默认 ``None`` 表示 ``ThreadPoolExecutor`` 的默认值。
    :return: 与 ``query_params`` 顺序一致的 DataFrame 序列。
    """

    tasks = []
    for index, (query, params) in enumerate(query_params):
        tasks.append((index, (pool, query), {"params": params}))
    return _run_with_thread_pool(
        lambda pool_obj, sql, params=None: read_dataframe(pool_obj, sql, params=params),
        tasks,
        max_workers=max_workers,
    )


def threaded_insert_dataframe(
    pool: DatabaseConnectionPool,
    datasets: Sequence[Mapping[str, Any]],
    max_workers: Optional[int] = None,
) -> None:
    """使用线程池并发插入多个 :class:`pandas.DataFrame`。

    :param pool: 连接池实例。
    :param datasets: 每个元素需至少包含 ``dataframe``、``table`` 键，可额外提供
        :func:`insert_dataframe` 支持的其他关键字参数（如 ``schema``、``replace`` 等）。
    :param max_workers: 线程池最大线程数。
    """

    def _worker(pool_obj: DatabaseConnectionPool, kwargs: Mapping[str, Any]) -> None:
        params = dict(kwargs)
        dataframe = params.pop("dataframe")
        table = params.pop("table")
        insert_dataframe(pool_obj, dataframe, table, **params)

    tasks = []
    for index, item in enumerate(datasets):
        tasks.append((index, (pool, item), {}))
    _run_with_thread_pool(
        lambda pool_obj, params: _worker(pool_obj, params),
        tasks,
        max_workers=max_workers,
    )


def threaded_update_dataframe(
    pool: DatabaseConnectionPool,
    datasets: Sequence[Mapping[str, Any]],
    max_workers: Optional[int] = None,
) -> None:
    """使用线程池并发更新多个 :class:`pandas.DataFrame`。

    :param pool: 连接池实例。
    :param datasets: 每个元素需包含 ``dataframe``、``table``、``key_columns`` 键，可附带
        :func:`update_dataframe` 支持的其他关键字参数（如 ``schema``、``batch_size`` 等）。
    :param max_workers: 线程池最大线程数。
    """

    def _worker(pool_obj: DatabaseConnectionPool, kwargs: Mapping[str, Any]) -> None:
        params = dict(kwargs)
        dataframe = params.pop("dataframe")
        table = params.pop("table")
        key_columns = params.pop("key_columns")
        update_dataframe(pool_obj, dataframe, table, key_columns=key_columns, **params)

    tasks = []
    for index, item in enumerate(datasets):
        tasks.append((index, (pool, item), {}))
    _run_with_thread_pool(
        lambda pool_obj, params: _worker(pool_obj, params),
        tasks,
        max_workers=max_workers,
    )


__all__ = [
    "create_connection",
    "DatabaseConnectionPool",
    "read_dataframe",
    "insert_dataframe",
    "update_dataframe",
    "threaded_read_dataframe",
    "threaded_insert_dataframe",
    "threaded_update_dataframe",
]
