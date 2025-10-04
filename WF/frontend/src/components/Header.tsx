import { Link, useLocation } from 'react-router-dom';

export default function Header() {
  const { pathname } = useLocation();
  const tabClass = (active: boolean) =>
    `px-3 py-2 text-sm rounded ${active ? 'bg-indigo-600 text-white' : 'text-indigo-700 hover:bg-indigo-50'}`;

  return (
    <header className="bg-white shadow">
      <div className="container mx-auto px-4 py-3 flex items-center gap-3">
        <div className="text-lg font-semibold text-gray-800">择时分析</div>
        <nav className="flex items-center gap-2">
          <Link to="/" className={tabClass(pathname === '/' || pathname.startsWith('/stocks'))}>股票列表</Link>
          <Link to="/fetch-all" className={tabClass(pathname.startsWith('/fetch-all'))}>批量股票数据获取</Link>
          <Link to="/full-abc" className={tabClass(pathname.startsWith('/full-abc'))}>批量股票ABC择时分析</Link>
        </nav>
      </div>
    </header>
  );
}
