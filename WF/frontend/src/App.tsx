import { BrowserRouter, Routes, Route } from 'react-router-dom';
import StockList from './pages/StockList';
import KLineChart from './pages/KLineChart';
import FullFetch from './pages/FullFetch';
import FullABC from './pages/FullABC';
import Header from './components/Header';

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-100">
        <Header />
        <main>
          <Routes>
            <Route path="/" element={<StockList />} />
            <Route path="/stocks/:symbol" element={<KLineChart />} />
            <Route path="/fetch-all" element={<FullFetch />} />
            <Route path="/full-abc" element={<FullABC />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
