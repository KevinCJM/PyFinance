import { useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

interface JobStatus {
  job_id: string;
  status: 'running'|'finished'|'error'|'cancelled';
  started_at?: string | null;
  finished_at?: string | null;
  total: number;
  done: number;
  success: number;
  fail: number;
  failed?: string[];
  logs: string[];
  last_symbol?: string | null;
  error?: string;
}

export default function FullFetch() {
  const [jobId, setJobId] = useState<string>('');
  const [status, setStatus] = useState<JobStatus | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);
  const nav = useNavigate();
  // 过滤参数
  const [symbolsText, setSymbolsText] = useState('');
  const [market, setMarket] = useState('');
  // 并发与限频
  const [maxWorkers, setMaxWorkers] = useState<number>(Math.max(1, navigator.hardwareConcurrency ? navigator.hardwareConcurrency - 1 : 4));
  const [maxCps, setMaxCps] = useState<number>(8);
  const [resume, setResume] = useState<boolean>(true);
  const [force, setForce] = useState<boolean>(false);
  const [refreshingBasic, setRefreshingBasic] = useState(false);

  // 进入页面时，若已有运行中的任务，则直接接管显示
  useEffect(() => {
    if (jobId) return;
    const probe = async () => {
      try {
        const r = await fetch('/api/fetch_all/active');
        if (!r.ok) return;
        const j = await r.json();
        if (j && j.job_id) {
          setJobId(j.job_id);
          setStatus(j);
        }
      } catch {}
    };
    probe();
  }, [jobId]);

  useEffect(() => {
    if (!jobId) return;
    let timer: any;
    const poll = async () => {
      try {
        const r = await fetch(`/api/fetch_all/${jobId}/status`);
        if (!r.ok) throw new Error('获取状态失败');
        const st: JobStatus = await r.json();
        setStatus(st);
        // 滚动到底部
        if (logRef.current) {
          logRef.current.scrollTop = logRef.current.scrollHeight;
        }
        if (st.status === 'running') {
          timer = setTimeout(poll, 1000);
        }
      } catch (e) {
        console.error(e);
        timer = setTimeout(poll, 1500);
      } finally { }
    };
    poll();
    return () => timer && clearTimeout(timer);
  }, [jobId]);

  const percent = status && status.total > 0 ? Math.floor((status.done / status.total) * 100) : 0;

  return (
    <div className="container mx-auto p-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">全量获取所有股票数据</h1>
        <div className="flex items-center gap-3">
          <button
            className="px-3 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded disabled:opacity-50"
            disabled={!!jobId || refreshingBasic}
            onClick={async () => {
              try {
                setRefreshingBasic(true);
                const r = await fetch('/api/stocks_basic/refresh', { method: 'POST' });
                if (!r.ok) throw new Error('刷新失败');
                await r.json();
              } catch (e) {
                alert((e as any).message || '刷新失败');
              } finally {
                setRefreshingBasic(false);
              }
            }}
          >{refreshingBasic ? '刷新中...' : '刷新股票列表'}</button>
          <Link to="/" className="text-indigo-600 hover:text-indigo-900">返回股票列表</Link>
        </div>
      </div>

      {!jobId && (
        <div className="mt-4 bg-white rounded shadow p-4 space-y-3">
          <div className="text-sm text-gray-700">批量股票数据获取：可指定股票或板块进行拉取。</div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <div className="font-medium">指定股票（逗号/空白分隔，多只）</div>
              <textarea className="mt-1 w-full border rounded p-2 h-24" placeholder="示例：000001, 600000, 300008" value={symbolsText} onChange={e => setSymbolsText(e.target.value)} />
            </div>
            <div>
              <div className="font-medium">指定板块</div>
              <select className="mt-1 w-full border rounded p-2" value={market} onChange={e => setMarket(e.target.value)}>
                <option value="">不指定</option>
                <option value="主板">主板</option>
                <option value="科创板">科创板</option>
                <option value="创业板">创业板</option>
                <option value="北交所">北交所</option>
              </select>
              <div className="text-xs text-gray-500 mt-1">若同时指定股票与板块，则仅对交集执行。</div>
            </div>
            <div>
              <div className="font-medium">最大并发线程数</div>
              <input className="mt-1 w-full border rounded p-2" type="number" value={maxWorkers} onChange={e => setMaxWorkers(Math.max(1, parseInt(e.target.value||'1',10)))} />
            </div>
            <div>
              <div className="font-medium">每秒最大调用次数（限频）</div>
              <input className="mt-1 w-full border rounded p-2" type="number" step="0.1" value={maxCps} onChange={e => setMaxCps(Math.max(0.1, parseFloat(e.target.value||'8')))} />
            </div>
          </div>
          <div className="text-xs text-gray-600">增量与重抓：默认增量（自动断点续跑）；如果发现历史异常可勾选“强制重抓”。</div>
          <div className="text-sm flex items-center gap-4">
            <label className="inline-flex items-center gap-2"><input type="checkbox" checked={resume} onChange={e => setResume(e.target.checked)} /><span>增量续跑</span></label>
            <label className="inline-flex items-center gap-2"><input type="checkbox" checked={force} onChange={e => setForce(e.target.checked)} /><span>强制重抓</span></label>
          </div>
          <div>
            <button className="px-3 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded" onClick={async () => {
              try {
                const symbols = Array.from(new Set(symbolsText.split(/[\s,]+/).map(s => s.trim()).filter(Boolean)));
                const payload: any = {};
                if (symbols.length) payload.symbols = symbols;
                if (market) payload.market = market;
                payload.max_workers = maxWorkers;
                payload.max_calls_per_second = maxCps;
                payload.resume = resume;
                payload.force = force;
                const r = await fetch('/api/fetch_all/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
                if (!r.ok) throw new Error('启动任务失败');
                const data = await r.json();
                setJobId(data.job_id);
              } catch (e) {
                alert((e as any).message || '启动失败');
              }
            }} disabled={refreshingBasic}>开始获取</button>
          </div>
        </div>
      )}

      {jobId && (
        <div className="mt-4 space-y-3">
          <div className="text-sm text-gray-700">任务ID：{jobId}</div>
          <div className="w-full bg-gray-200 rounded h-2 overflow-hidden">
            <div className="bg-emerald-500 h-2" style={{ width: `${percent}%` }} />
          </div>
          <div className="text-sm text-gray-700">
            状态：{status?.status || '启动中...'}，进度：{status?.done || 0}/{status?.total || 0}，成功：{status?.success || 0}，失败：{status?.fail || 0}
          </div>

          {status?.error && (
            <div className="text-sm text-red-600">错误：{status.error}</div>
          )}

          <div>
            <div className="text-sm font-medium mb-1">日志</div>
            <div ref={logRef} className="border rounded h-72 overflow-auto text-xs p-2 bg-white">
              {(status?.logs || []).map((line, idx) => (
                <div key={idx} className="whitespace-pre-wrap">{line}</div>
              ))}
            </div>
          </div>

          {status?.status === 'finished' && (
            <div className="flex items-center gap-3">
              <button className="px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded" onClick={() => nav('/')}>返回</button>
              {status?.failed && status.failed.length > 0 && (
                <div className="text-sm text-gray-700">失败代码：{status.failed.join(', ')}</div>
              )}
            </div>
          )}
        </div>
      )}

      {(refreshingBasic || jobId) && (
        <div className="fixed inset-0 pointer-events-none">
          {refreshingBasic && (
            <div className="fixed inset-0 bg-black/30 z-50 flex items-center justify-center pointer-events-auto">
              <div className="bg-white rounded px-6 py-4 shadow text-sm">正在刷新股票列表，请稍候...</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
