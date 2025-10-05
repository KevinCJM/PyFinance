import { useState, useEffect, useMemo } from 'react';
import { Link } from 'react-router-dom';

interface Stock {
  symbol: string;
  name: string;
  fullname: string;
  market: string;
  exchange: string;
  area: string;
  industry: string;
  list_date: string;
  list_status: string;
}

export default function StockList() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshingBasic, setRefreshingBasic] = useState(false);
  const [error, setError] = useState('');
  const [version, setVersion] = useState(0); // 触发刷新
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [total, setTotal] = useState(0);
  const [sortBy, setSortBy] = useState('symbol');
  const [sortDir, setSortDir] = useState('asc');
  const [marketFilter, setMarketFilter] = useState('');
  const [industryFilter, setIndustryFilter] = useState('');
  const [codeQuery, setCodeQuery] = useState('');
  const [nameQuery, setNameQuery] = useState('');
  const [nameFuzzy, setNameFuzzy] = useState(true);

  useEffect(() => {
    const fetchStocks = async () => {
      setLoading(true);
      try {
        const params = new URLSearchParams({
          page: String(page),
          page_size: String(pageSize),
          sort_by: sortBy,
          sort_dir: sortDir,
          market: marketFilter,
          industry: industryFilter,
        });
        if (codeQuery) params.set('code', codeQuery);
        if (nameQuery) {
          params.set('name', nameQuery);
          params.set('name_fuzzy', String(nameFuzzy));
          // 可选：调节阈值，默认后端 0.55，这里保持默认不传
        }
        const res = await fetch(`/api/stocks?${params.toString()}`);
        if (!res.ok) {
          throw new Error('Failed to fetch stocks');
        }
        const data = await res.json();
        setStocks(data.items);
        setTotal(data.total);
      } catch (e: any) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };

    fetchStocks();
  }, [page, pageSize, sortBy, sortDir, marketFilter, industryFilter, codeQuery, nameQuery, nameFuzzy, version]);

  const handleSort = (column: string) => {
    if (sortBy === column) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setSortDir('asc');
    }
  };

  const totalPages = useMemo(() => Math.ceil(total / pageSize), [total, pageSize]);

  return (
    <div className="container mx-auto p-4">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-2xl font-bold">股票列表</h1>
        <button
          className="px-3 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded disabled:opacity-50"
          disabled={refreshingBasic}
          onClick={async () => {
            try {
              setRefreshingBasic(true);
              const r = await fetch('/api/stocks_basic/refresh', { method: 'POST' });
              if (!r.ok) throw new Error('刷新失败');
              await r.json();
              // 重新加载列表
              setVersion(v => v + 1);
            } catch (e: any) {
              setError(e.message || '刷新失败');
            } finally {
              setRefreshingBasic(false);
            }
          }}
        >{refreshingBasic ? '刷新中...' : '刷新股票列表'}</button>
      </div>

      <div className="flex space-x-4 mb-4">
        <div>
          <label htmlFor="code-filter" className="block text-sm font-medium text-gray-700">股票代码</label>
          <input
            id="code-filter"
            type="text"
            value={codeQuery}
            onChange={(e) => setCodeQuery(e.target.value)}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
            placeholder="例如: 300008"
          />
        </div>
        <div>
          <label htmlFor="name-filter" className="block text-sm font-medium text-gray-700">股票名称</label>
          <input
            id="name-filter"
            type="text"
            value={nameQuery}
            onChange={(e) => setNameQuery(e.target.value)}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
            placeholder="支持相似度搜索，例如: 天海防务"
          />
          <div className="mt-1 text-xs text-gray-600">
            <label className="inline-flex items-center space-x-1">
              <input type="checkbox" checked={nameFuzzy} onChange={e => setNameFuzzy(e.target.checked)} />
              <span>启用相似度匹配</span>
            </label>
          </div>
        </div>
        <div>
          <label htmlFor="market-filter" className="block text-sm font-medium text-gray-700">市场类型</label>
          <input
            id="market-filter"
            type="text"
            value={marketFilter}
            onChange={(e) => setMarketFilter(e.target.value)}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
            placeholder="例如: 主板"
          />
        </div>
        <div>
          <label htmlFor="industry-filter" className="block text-sm font-medium text-gray-700">所属行业</label>
          <input
            id="industry-filter"
            type="text"
            value={industryFilter}
            onChange={(e) => setIndustryFilter(e.target.value)}
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
            placeholder="例如: 银行"
          />
        </div>
      </div>

      {loading && <p>加载中...</p>}
      {error && <p className="text-red-500">{error}</p>}
      
      {!loading && !error && (
        <>
          {refreshingBasic && (
            <div className="fixed inset-0 bg-black/30 z-50 flex items-center justify-center">
              <div className="bg-white rounded px-6 py-4 shadow text-sm">正在刷新股票列表，请稍候...</div>
            </div>
          )}
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  {['symbol', 'name', 'market', 'industry', 'list_date'].map((col) => (
                    <th key={col} scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer" onClick={() => handleSort(col)}>
                      {col} {sortBy === col && (sortDir === 'asc' ? '▲' : '▼')}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {stocks.map((stock) => (
                  <tr key={stock.symbol}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      <Link to={`/stocks/${stock.symbol}`} className="text-indigo-600 hover:text-indigo-900">{stock.symbol}</Link>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <Link to={`/stocks/${stock.symbol}`} className="text-indigo-600 hover:text-indigo-900">{stock.name}</Link>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{stock.market}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{stock.industry}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{stock.list_date}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex items-center justify-between mt-4">
            <div>
              <span className="text-sm text-gray-700">
                第 {page} 页 / 共 {totalPages} 页 (总计 {total} 条)
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <select value={pageSize} onChange={e => setPageSize(Number(e.target.value))} className="px-2 py-1 text-sm border-gray-300 rounded-md">
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
              <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1} className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50">
                上一页
              </button>
              <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page === totalPages} className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50">
                下一页
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
