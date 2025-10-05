import { useEffect, useRef, useState } from 'react';
import AParamsPanel from '../components/AParamsPanel';
import BParamsPanel from '../components/BParamsPanel';
import CParamsPanel from '../components/CParamsPanel';

interface JobStatus {
  job_id: string;
  status: 'running'|'finished'|'error'|'cancelled'|'unknown';
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
  b_recent?: { symbol: string; name?: string; market?: string }[];
}

export default function FullABC() {
  // 过滤参数
  const [symbolsText, setSymbolsText] = useState('');
  const [market, setMarket] = useState('');
  const [maxWorkers, setMaxWorkers] = useState<number>(Math.max(1, navigator.hardwareConcurrency ? navigator.hardwareConcurrency - 1 : 4));

  // A 点参数（与个股页一致）
  const [aCond1, setACond1] = useState({ 启用: true, 长均线窗口: 60, 下跌跨度: 30 });
  const [aCond2, setACond2] = useState({ 启用: true, 短均线集合: '5,10', 长均线窗口: 60, 上穿完备窗口: 3, 必须满足的短均线: '', 全部满足: false });
  const [aCond3, setACond3] = useState({ 启用: false, 确认回看天数: 0, 确认均线窗口: '', 确认价格列: 'high' });
  const [aCondVol, setACondVol] = useState({ 启用: true, vr1_enabled: false, 对比天数: 10, 倍数: 2.0, vma_cmp_enabled: true, 短期天数: 5, 长期天数: 10, vol_up_enabled: true, 量连升天数: 3 });

  // B 点参数
  const [bCond1, setBCond1] = useState({ enabled: true, min_days_from_a: 60, max_days_from_a: '' as number|'', allow_multi_b_per_a: true });
  const [bCond2MA, setBCond2MA] = useState({ enabled: true, above_maN_window: 5, above_maN_days: 15, above_maN_consecutive: false, max_maN_below_days: 5, long_ma_days: 60, above_maN_ratio: 60 });
  const [bCond2, setBCond2] = useState({ enabled: true, touch_price: 'low' as 'low'|'close', touch_relation: 'le' as 'le'|'lt', require_bearish: false, require_close_le_prev: false, long_ma_days: 60 });
  const [bCond4VR, setBCond4VR] = useState({ enabled: false, vr1_max: '' as number|'', recent_max_vol_window: 10 });
  const [bCond3, setBCond3] = useState({ enabled: true, dryness_ratio_max: 0.8, require_vol_le_vma10: true, dryness_recent_window: 0, dryness_recent_min_days: 0, short_days: 5, long_days: 10, vol_compare_long_window: 10, vr1_enabled: false, vma_rel_enabled: true, vol_down_enabled: true, vol_decreasing_days: 3 });
  const [bCond4, setBCond4] = useState({ enabled: false, price_stable_mode: 'no_new_low' as 'no_new_low'|'ratio'|'atr', max_drop_ratio: 0.03, use_atr_window: 14, atr_buffer: 0.5 });

  // C 点参数
  const [cCond1, setCCond1] = useState({ enabled: true, max_days_from_b: 60 });
  const [cCond2, setCCond2] = useState({ enabled: true, vr1_enabled: false, recent_n: 10, vol_multiple: 2.0, vma_cmp_enabled: false, vma_short_days: 5, vma_long_days: 10, vol_up_enabled: true, vol_increasing_days: 3 });
  const [cCond3, setCCond3] = useState({ enabled: true, price_field: 'close' as 'close'|'high'|'low', ma_days: 60, relation: 'ge' as 'ge'|'gt' });

  // 任务
  const [jobId, setJobId] = useState('');
  const [status, setStatus] = useState<JobStatus | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);

  // 轮询
  useEffect(() => {
    if (!jobId) return;
    let timer: any;
    const poll = async () => {
      try {
        const r = await fetch(`/api/abc_batch/${jobId}/status`);
        if (!r.ok) return;
        const st: JobStatus = await r.json();
        setStatus(st);
        if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
        if (st.status === 'running') timer = setTimeout(poll, 1000);
      } catch {}
    };
    poll();
    return () => timer && clearTimeout(timer);
  }, [jobId]);

  // 页面进入接管活跃任务
  useEffect(() => {
    if (jobId) return;
    (async () => {
      try {
        const r = await fetch('/api/abc_batch/active');
        if (!r.ok) return;
        const j = await r.json();
        if (j && j.job_id) { setJobId(j.job_id); setStatus(j); }
      } catch {}
    })();
  }, [jobId]);

  const percent = status && status.total > 0 ? Math.floor((status.done / status.total) * 100) : 0;

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">批量股票ABC择时分析</h1>

      {/* 选择范围 */}
      <div className="bg-white p-4 rounded shadow mb-4">
        <div className="font-semibold mb-2">选择范围</div>
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
            <div className="font-medium">最大并发进程数</div>
            <input type="number" className="mt-1 w-full border rounded p-2" value={maxWorkers} onChange={e => setMaxWorkers(Math.max(1, parseInt(e.target.value||'1',10)))} />
            <div className="text-xs text-gray-500 mt-1">默认 CPU 数量-1；建议不要超过物理核心数。</div>
          </div>
        </div>
      </div>

      {/* A/B/C 参数（完全复用个股页的面板） */}
      <AParamsPanel aCond1={aCond1} setACond1={setACond1} aCond2={aCond2} setACond2={setACond2} aCond3={aCond3} setACond3={setACond3} aCondVol={aCondVol} setACondVol={setACondVol} computingA={false} />
      <BParamsPanel bCond1={bCond1} setBCond1={setBCond1} bCond2MA={bCond2MA} setBCond2MA={setBCond2MA} bCond2={bCond2} setBCond2={setBCond2} bCond4VR={bCond4VR} setBCond4VR={setBCond4VR} bCond3={bCond3} setBCond3={setBCond3} bCond4={bCond4} setBCond4={setBCond4} computingB={false} />
      <CParamsPanel cCond1={cCond1} setCCond1={setCCond1} cCond2={cCond2} setCCond2={setCCond2} cCond3={cCond3} setCCond3={setCCond3} computingC={false} />

      {/* 操作与进度 */}
      <div className="bg-white p-4 rounded shadow">
        <button
          className="px-3 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded disabled:opacity-50"
          onClick={async () => {
            try {
              const payload: any = {
                a_params: { '条件1_长期下跌': aCond1, '条件2_短均线上穿': aCond2, '条件3_价格确认': aCond3, '条件4_放量确认': aCondVol },
                b_params: { cond1: bCond1, cond2: bCond2MA, cond3: bCond2, cond4: { enabled: bCond4VR.enabled, vr1_max: bCond4VR.vr1_max === '' ? null : bCond4VR.vr1_max, recent_max_vol_window: bCond4VR.recent_max_vol_window }, cond5: { ...bCond3, vma_short_window: bCond3.short_days, vma_long_window: bCond3.long_days, vol_compare_long_window: bCond3.vol_compare_long_window }, cond6: bCond4 },
                c_params: { cond1: cCond1, cond2: cCond2, cond3: cCond3 },
                max_workers: maxWorkers,
              };
              const symbols = Array.from(new Set(symbolsText.split(/[\s,]+/).map(s => s.trim()).filter(Boolean)));
              if (symbols.length) payload.symbols = symbols;
              if (market) payload.market = market;
              const r = await fetch('/api/abc_batch/start', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
              if (!r.ok) throw new Error('启动失败');
              const data = await r.json();
              setJobId(data.job_id);
            } catch (e: any) {
              alert(e.message || '启动失败');
            }
          }}
          disabled={status?.status === 'running'}
        >开始批量分析</button>

        {status && (
          <div className="mt-4 space-y-3">
            <div className="w-full bg-gray-200 rounded h-2 overflow-hidden"><div className="bg-emerald-500 h-2" style={{ width: `${percent}%` }} /></div>
            <div className="text-sm text-gray-700">状态：{status.status}，进度：{status.done}/{status.total}，成功：{status.success}，失败：{status.fail}</div>
            <div>
              <div className="text-sm font-medium mb-1">日志</div>
              <div ref={logRef} className="border rounded h-64 overflow-auto text-xs p-2 bg-gray-50">{(status.logs || []).map((line, idx) => (<div key={idx} className="whitespace-pre-wrap">{line}</div>))}</div>
            </div>
            <div>
              <div className="text-sm font-medium mb-1">近5天出现B点的股票</div>
              <div className="overflow-auto">
                <table className="min-w-[480px] table-fixed border-collapse">
                  <thead className="bg-gray-50 text-xs"><tr><th className="border px-2 py-1 text-left">股票代码</th><th className="border px-2 py-1 text-left">股票名称</th><th className="border px-2 py-1 text-left">所属市场</th></tr></thead>
                  <tbody className="text-xs">
                    {(status.b_recent || []).map((r, i) => (<tr key={i}><td className="border px-2 py-1">{r.symbol}</td><td className="border px-2 py-1">{r.name || '-'}</td><td className="border px-2 py-1">{r.market || '-'}</td></tr>))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

