import { useEffect, useRef, useState } from 'react';
import AParamsPanel from '../components/AParamsPanel';
import BParamsPanel from '../components/BParamsPanel';
import CParamsPanel from '../components/CParamsPanel';
import KLineEChart from '../components/KLineEChart';
import { useABCDefaults } from '../hooks/useABCDefaults';

interface JobStatus {
  job_id: string;
  status: 'running'|'canceling'|'finished'|'error'|'cancelled'|'unknown';
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
  b_recent?: { symbol: string; name?: string; industry?: string; market?: string; last_b_date?: string; first_c_date?: string; last_c_date?: string }[];
  b_recent_days?: number;
}

export default function FullABC() {
  const { aDefaults, bDefaults, cDefaults } = useABCDefaults();
  // 过滤参数
  const [symbolsText, setSymbolsText] = useState('');
  const [market, setMarket] = useState('');
  const [maxWorkers, setMaxWorkers] = useState<number>(Math.max(1, navigator.hardwareConcurrency ? navigator.hardwareConcurrency - 1 : 4));

  const [bRecentDays, setBRecentDays] = useState<number>(5);

  // A 点参数（与个股页一致）
  const [aCond1, setACond1] = useState(aDefaults.aCond1);
  const [aCond2, setACond2] = useState(aDefaults.aCond2);
  const [aCond3, setACond3] = useState(aDefaults.aCond3);
  const [aCondVol, setACondVol] = useState(aDefaults.aCondVol);

  // B 点参数
  const [bCond1, setBCond1] = useState(bDefaults.bCond1);
  const [bCond2MA, setBCond2MA] = useState(bDefaults.bCond2MA);
  const [bCond2, setBCond2] = useState(bDefaults.bCond2);
  const [bCond4VR, setBCond4VR] = useState(bDefaults.bCond4VR);
  const [bCond3, setBCond3] = useState(bDefaults.bCond3);
  const [bCond4, setBCond4] = useState(bDefaults.bCond4);

  // C 点参数
  const [cCond1, setCCond1] = useState(cDefaults.cCond1);
  const [cCond2, setCCond2] = useState(cDefaults.cCond2);
  const [cCond3, setCCond3] = useState(cDefaults.cCond3);

  // 任务
  const [jobId, setJobId] = useState('');
  const [status, setStatus] = useState<JobStatus | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);
  // 预览选中的股票
  const [selectedSymbol, setSelectedSymbol] = useState<string>('');
  const [previewKline, setPreviewKline] = useState<any[]>([]);
  const [previewA, setPreviewA] = useState<any[]>([]);
  const [previewB, setPreviewB] = useState<any[]>([]);
  const [previewC, setPreviewC] = useState<any[]>([]);
  const [previewLoading, setPreviewLoading] = useState<boolean>(false);
  const [previewError, setPreviewError] = useState<string>('');


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
        if (st.status === 'running' || st.status === 'canceling') timer = setTimeout(poll, 1000);
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
          <div>
            <div className="font-medium">统计近 N 天出现B点的股票</div>
            <input type="number" min={1} className="mt-1 w-full border rounded p-2" value={bRecentDays} onChange={e => setBRecentDays(Math.max(1, parseInt(e.target.value||'1',10)))} />
            <div className="text-xs text-gray-500 mt-1">该参数仅影响下方“近N天出现B点的股票”列表。</div>
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
                b_params: {
                  cond1: bCond1,
                  // 与单股页面保持一致的 cond2 映射（long_ma_days -> long_ma_window，比例由百分比转为 0-1）
                  cond2: (() => {
                    const mode = (bCond2MA as any).mode ?? 'ratio';
                    const base: any = {
                      enabled: bCond2MA.enabled,
                      above_maN_window: bCond2MA.above_maN_window,
                      max_maN_below_days: bCond2MA.max_maN_below_days,
                      long_ma_window: bCond2MA.long_ma_days,
                    };
                    if (mode === 'ratio') {
                      if (bCond2MA.above_maN_ratio != null) base.above_maN_ratio = (bCond2MA.above_maN_ratio / 100);
                    } else {
                      base.above_maN_days = bCond2MA.above_maN_days;
                      base.above_maN_consecutive = bCond2MA.above_maN_consecutive;
                    }
                    return base;
                  })(),
                  cond3: bCond2,
                  cond4: { enabled: bCond4VR.enabled, vr1_max: bCond4VR.vr1_max === '' ? null : bCond4VR.vr1_max, recent_max_vol_window: bCond4VR.recent_max_vol_window },
                  cond5: { ...bCond3, vma_short_window: bCond3.short_days, vma_long_window: bCond3.long_days, vol_compare_long_window: bCond3.vol_compare_long_window },
                  cond6: bCond4,
                },
                c_params: { cond1: cCond1, cond2: cCond2, cond3: cCond3 },
                max_workers: maxWorkers,
                b_recent_days: bRecentDays,
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

        <button
          className="ml-2 px-3 py-2 bg-red-600 text-white rounded disabled:opacity-50"
          onClick={async () => {
            try {
              const targetJobId = jobId || (await (await fetch('/api/abc_batch/active')).json()).job_id;
              if (!targetJobId) return;
              const r = await fetch(`/api/abc_batch/${targetJobId}/stop`, { method: 'POST' });
              if (!r.ok) throw new Error('停止失败');
              const st = await r.json();
              setStatus((prev) => ({...(prev as any), ...(st||{}), job_id: targetJobId }));
            } catch (e: any) {
              alert(e.message || '停止失败');
            }
          }}
          disabled={!(status?.status === 'running' && jobId)}
        >停止计算</button>

        {status && (
          <div className="mt-4 space-y-3">
            <div className="w-full bg-gray-200 rounded h-2 overflow-hidden"><div className="bg-emerald-500 h-2" style={{ width: `${percent}%` }} /></div>
            <div className="text-sm text-gray-700">状态：{status.status}，进度：{status.done}/{status.total}，成功：{status.success}，失败：{status.fail}</div>
            <div>
              <div className="text-sm font-medium mb-1">日志</div>
              <div ref={logRef} className="border rounded h-64 overflow-auto text-xs p-2 bg-gray-50">{(status.logs || []).map((line, idx) => (<div key={idx} className="whitespace-pre-wrap">{line}</div>))}</div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <div className="text-sm font-medium">近{(status?.b_recent_days ?? bRecentDays)}天出现B点的股票（点击代码/名称查看K线预览）</div>
                <button
                  className="px-2 py-1 text-xs border rounded hover:bg-gray-50"
                  onClick={() => {
                    const rows = (status?.b_recent || []).map((r: any) => [r.symbol, r.name || '', r.industry || '', r.market || '', (r as any).last_b_date || '', (r as any).first_c_date || '', (r as any).last_c_date || '']);
const headers = ['股票代码','股票名称','所属行业','所属市场','最后的B点日期', '首次C点', '末次C点'];
const lines = rows.map((r: any) => r.map((x: any) => {
  const s = String(x ?? '');
  return (s.includes(',') || s.includes('"') || s.includes('\n')) ? "" + s.replace(/"/g,'""') + "" : s;
}).join(','));
const csv = '\ufeff' + [headers.join(','), ...lines].join('\n');
const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
const url = URL.createObjectURL(blob);
const a = document.createElement('a');
a.href = url;
a.download = `近${(status?.b_recent_days ?? bRecentDays)}天_B点股票列表.csv`;
a.click();
URL.revokeObjectURL(url)
                  }}
                >导出CSV</button>
              </div>
              <div className="overflow-auto h-96">
                <table className="min-w-[640px] table-fixed border-collapse">
                  <thead className="bg-gray-50 text-xs"><tr><th className="border px-2 py-1 text-left">股票代码</th><th className="border px-2 py-1 text-left">股票名称</th><th className="border px-2 py-1 text-left">所属行业</th><th className="border px-2 py-1 text-left">所属市场</th><th className="border px-2 py-1 text-left">最后的B点日期</th><th className="border px-2 py-1 text-left">首次C点</th><th className="border px-2 py-1 text-left">末次C点</th></tr></thead>
                  <tbody className="text-xs">
                    {(status.b_recent || []).map((r, i) => (
                    <tr key={i}>
                      <td className="border px-2 py-1">
                        <button data-sym={String(r.symbol)} className="text-indigo-600 hover:underline" onClick={async () => {
                          const sym = String(r.symbol);
                          setSelectedSymbol(sym); setPreviewError(''); setPreviewLoading(true);
                          try {
                            const kd = await fetch(`/api/stocks/${sym}`); if (!kd.ok) throw new Error('获取K线失败'); const k = await kd.json(); setPreviewKline(k);
                            const aBody = { '条件1_长期下跌': aCond1, '条件2_短均线上穿': aCond2, '条件3_价格确认': aCond3, '条件4_放量确认': aCondVol };
                            const ar = await fetch(`/api/stocks/${sym}/a_points_v2`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(aBody) }); if (!ar.ok) throw new Error('A点计算失败'); const ajson = await ar.json(); const apts = Array.isArray(ajson.points)? ajson.points: []; setPreviewA(apts);
                            const bBody: any = {
                              cond1: bCond1,
                              cond2: (() => {
                                const mode = (bCond2MA as any).mode ?? 'ratio';
                                const base: any = {
                                  enabled: bCond2MA.enabled,
                                  above_maN_window: bCond2MA.above_maN_window,
                                  max_maN_below_days: bCond2MA.max_maN_below_days,
                                  long_ma_window: bCond2MA.long_ma_days,
                                };
                                if (mode === 'ratio') {
                                  if (bCond2MA.above_maN_ratio != null) base.above_maN_ratio = (bCond2MA.above_maN_ratio / 100);
                                } else {
                                  base.above_maN_days = bCond2MA.above_maN_days;
                                  base.above_maN_consecutive = bCond2MA.above_maN_consecutive;
                                }
                                return base;
                              })(),
                              cond3: bCond2,
                              cond4: { enabled: bCond4VR.enabled, vr1_max: bCond4VR.vr1_max === '' ? null : bCond4VR.vr1_max, recent_max_vol_window: bCond4VR.recent_max_vol_window },
                              cond5: { ...bCond3, vma_short_window: bCond3.short_days, vma_long_window: bCond3.long_days, vol_compare_long_window: bCond3.vol_compare_long_window },
                              cond6: bCond4,
                              a_points_dates: apts.map((p:any)=>p.date),
                            };
                            const br = await fetch(`/api/stocks/${sym}/b_points`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(bBody) }); if (!br.ok) throw new Error('B点计算失败'); const bjson = await br.json(); const bpts = Array.isArray(bjson.points)? bjson.points: []; setPreviewB(bpts);
                            const cBody: any = { cond1: cCond1, cond2: cCond2, cond3: cCond3, a_points_dates: apts.map((p:any)=>p.date), b_points_dates: bpts.map((p:any)=>p.date) };
                            const cr = await fetch(`/api/stocks/${sym}/c_points`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cBody) }); if (!cr.ok) throw new Error('C点计算失败'); const cjson = await cr.json(); setPreviewC(Array.isArray(cjson.points)? cjson.points: []);
                          } catch (e:any) { setPreviewError(e.message || '预览失败'); } finally { setPreviewLoading(false); }
                        }}>{r.symbol}</button>
                      </td>
                      <td className="border px-2 py-1">
                        <button className="text-indigo-600 hover:underline" onClick={() => { const btn = document.querySelector(`[data-sym="${String(r.symbol)}"]`) as HTMLButtonElement|null; btn?.click(); }}>{r.name || '-'}</button>
                      </td>
                      <td className="border px-2 py-1">{(r as any).industry || '-'}</td>
                      <td className="border px-2 py-1">{r.market || '-'}</td>
                      <td className="border px-2 py-1">{(r as any).last_b_date || '-'}</td>
                      <td className="border px-2 py-1">{(r as any).first_c_date || '-'}</td>
                      <td className="border px-2 py-1">{(r as any).last_c_date || '-'}</td>
                    </tr>
                  ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      {/* 选中股票的K线预览 */}
      {selectedSymbol && (
        <div className="mt-6 bg-white p-4 rounded shadow">
          <div className="flex items-center justify-between mb-2">
            <div className="font-medium">{selectedSymbol} K线预览（A/B/C点）</div>
            <div className="text-sm flex items-center gap-2">
              <button className="px-2 py-1 border rounded hover:bg-gray-50" onClick={() => { setSelectedSymbol(''); setPreviewKline([]); setPreviewA([]); setPreviewB([]); setPreviewC([]); }}>清除预览</button>
            </div>
          </div>
          {previewError && <div className="text-xs text-red-500 mb-2">{previewError}</div>}
          {previewLoading ? (
            <div className="h-64 flex items-center justify-center text-sm text-gray-600">加载与计算中...</div>
          ) : (
            <KLineEChart symbol={selectedSymbol} klineData={previewKline as any} aPoints={previewA as any} bPoints={previewB as any} cPoints={previewC as any} />
          )}
        </div>
      )}
      </div>
    </div>
  );
}
