import { useEffect, useRef, useState } from 'react';

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
  b_recent?: { symbol: string; name?: string; market?: string }[];
}

export default function FullABC() {
  // 过滤参数：指定股票/板块
  const [symbolsText, setSymbolsText] = useState('');
  const [market, setMarket] = useState('');
  // A 点参数（与个股页保持一致）
  const [aCond1, setACond1] = useState<{ 启用: boolean; 长均线窗口: number; 下跌跨度: number }>({ 启用: true, 长均线窗口: 60, 下跌跨度: 30 });
  const [aCond2, setACond2] = useState<{ 启用: boolean; 短均线集合: string; 长均线窗口: number; 上穿完备窗口: number; 必须满足的短均线: string; 全部满足: boolean }>({ 启用: true, 短均线集合: '5,10', 长均线窗口: 60, 上穿完备窗口: 3, 必须满足的短均线: '', 全部满足: false });
  const [aCond3, setACond3] = useState<{ 启用: boolean; 确认回看天数: number; 确认均线窗口: string; 确认价格列: string }>({ 启用: false, 确认回看天数: 0, 确认均线窗口: '', 确认价格列: 'high' });
  const [aCondVol, setACondVol] = useState<{ 启用: boolean; vr1_enabled: boolean; 对比天数: number; 倍数: number; vma_cmp_enabled: boolean; 短期天数: number; 长期天数: number; vol_up_enabled: boolean; 量连升天数: number }>(
    { 启用: true, vr1_enabled: false, 对比天数: 10, 倍数: 2.0, vma_cmp_enabled: true, 短期天数: 5, 长期天数: 10, vol_up_enabled: true, 量连升天数: 3 }
  );

  // B 点参数
  const [bCond1, setBCond1] = useState<{ enabled: boolean; min_days_from_a: number; max_days_from_a: number | ''; allow_multi_b_per_a: boolean }>(
    { enabled: true, min_days_from_a: 60, max_days_from_a: '', allow_multi_b_per_a: true }
  );
  const [bCond2MA, setBCond2MA] = useState<{ enabled: boolean; above_maN_window: number; above_maN_days: number; above_maN_consecutive: boolean; max_maN_below_days: number; long_ma_days: number; above_maN_ratio?: number }>(
    { enabled: true, above_maN_window: 5, above_maN_days: 15, above_maN_consecutive: false, max_maN_below_days: 5, long_ma_days: 60, above_maN_ratio: 60 }
  );
  const [bCond2, setBCond2] = useState<{ enabled: boolean; touch_price: 'low'|'close'; touch_relation: 'le'|'lt'; require_bearish: boolean; require_close_le_prev: boolean; long_ma_days: number }>(
    { enabled: true, touch_price: 'low', touch_relation: 'le', require_bearish: false, require_close_le_prev: false, long_ma_days: 60 }
  );
  const [bCond4VR, setBCond4VR] = useState<{ enabled: boolean; vr1_max: number | ''; recent_max_vol_window: number }>(
    { enabled: false, vr1_max: '', recent_max_vol_window: 10 }
  );
  const [bCond3, setBCond3] = useState<{ enabled: boolean; dryness_ratio_max: number; require_vol_le_vma10: boolean; dryness_recent_window: number; dryness_recent_min_days: number; short_days: number; long_days: number; vol_compare_long_window: number; vr1_enabled?: boolean; vma_rel_enabled?: boolean; vol_down_enabled?: boolean; vol_decreasing_days?: number }>(
    { enabled: true, dryness_ratio_max: 0.8, require_vol_le_vma10: true, dryness_recent_window: 0, dryness_recent_min_days: 0, short_days: 5, long_days: 10, vol_compare_long_window: 10, vr1_enabled: false, vma_rel_enabled: true, vol_down_enabled: true, vol_decreasing_days: 3 }
  );
  const [bCond4, setBCond4] = useState<{ enabled: boolean; price_stable_mode: 'no_new_low'|'ratio'|'atr'; max_drop_ratio: number; use_atr_window: number; atr_buffer: number }>(
    { enabled: false, price_stable_mode: 'no_new_low', max_drop_ratio: 0.03, use_atr_window: 14, atr_buffer: 0.5 }
  );

  // C 点参数
  const [cCond1, setCCond1] = useState<{ enabled: boolean; max_days_from_b: number }>({ enabled: true, max_days_from_b: 60 });
  const [cCond2, setCCond2] = useState<{ enabled: boolean; vr1_enabled: boolean; recent_n: number; vol_multiple: number; vma_cmp_enabled: boolean; vma_short_days: number; vma_long_days: number; vol_up_enabled: boolean; vol_increasing_days: number }>(
    { enabled: true, vr1_enabled: false, recent_n: 10, vol_multiple: 2.0, vma_cmp_enabled: false, vma_short_days: 5, vma_long_days: 10, vol_up_enabled: true, vol_increasing_days: 3 }
  );
  const [cCond3, setCCond3] = useState<{ enabled: boolean; price_field: 'close'|'high'|'low'; ma_days: number; relation: 'ge'|'gt' }>({ enabled: true, price_field: 'close', ma_days: 60, relation: 'ge' });

  // 任务
  const [jobId, setJobId] = useState<string>('');
  const [status, setStatus] = useState<JobStatus | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);
  const [maxWorkers, setMaxWorkers] = useState<number>(Math.max(1, navigator.hardwareConcurrency ? navigator.hardwareConcurrency - 1 : 4));

  useEffect(() => {
    if (!jobId) return;
    let timer: any;
    const poll = async () => {
      try {
        const r = await fetch(`/api/abc_batch/${jobId}/status`);
        if (!r.ok) throw new Error('获取状态失败');
        const st: JobStatus = await r.json();
        setStatus(st);
        if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
        if (st.status === 'running') timer = setTimeout(poll, 1000);
      } catch (e) {
        timer = setTimeout(poll, 1500);
      }
    };
    poll();
    return () => timer && clearTimeout(timer);
  }, [jobId]);

  // 进入页面时探测是否已有运行中的任务，若有则接管显示
  useEffect(() => {
    if (jobId) return;
    const probe = async () => {
      try {
        const r = await fetch('/api/abc_batch/active');
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

  const percent = status && status.total > 0 ? Math.floor((status.done / status.total) * 100) : 0;

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">批量股票ABC择时分析</h1>

      {/* 过滤选项 */}
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

      {/* A 点参数 */}
      <div className="bg-white p-4 rounded shadow mb-4">
        <div className="font-semibold mb-2">A 点参数</div>
        <div className="grid grid-cols-1 gap-3">
          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件1：长均线下跌</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={aCond1.启用} onChange={e => setACond1(p => ({...p, 启用: e.target.checked}))} /><span>启用</span></label>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mt-2">
              <div>
                <div>长均线窗口</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCond1.长均线窗口} onChange={e => setACond1(p => ({...p, 长均线窗口: parseInt(e.target.value||'60',10)}))} />
              </div>
              <div>
                <div>下跌跨度</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCond1.下跌跨度} onChange={e => setACond1(p => ({...p, 下跌跨度: parseInt(e.target.value||'30',10)}))} />
              </div>
            </div>
          </div>
          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件2：短均线上穿</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={aCond2.启用} onChange={e => setACond2(p => ({...p, 启用: e.target.checked}))} /><span>启用</span></label>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mt-2">
              <div>
                <div>短均线集合</div>
                <input type="text" className="mt-1 px-2 py-1 border rounded w-full" value={aCond2.短均线集合} onChange={e => setACond2(p => ({...p, 短均线集合: e.target.value}))} />
              </div>
              <div>
                <div>长均线窗口</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCond2.长均线窗口} onChange={e => setACond2(p => ({...p, 长均线窗口: parseInt(e.target.value||'60',10)}))} />
              </div>
              <div>
                <div>上穿完备窗口</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCond2.上穿完备窗口} onChange={e => setACond2(p => ({...p, 上穿完备窗口: parseInt(e.target.value||'3',10)}))} />
              </div>
              <div className="flex items-center gap-2">
                <input type="checkbox" className="mr-2" checked={aCond2.全部满足} onChange={e => setACond2(p => ({...p, 全部满足: e.target.checked}))} />
                <span>必须集合内均满足</span>
              </div>
            </div>
          </div>
          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件3：价格确认</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={aCond3.启用} onChange={e => setACond3(p => ({...p, 启用: e.target.checked}))} /><span>启用</span></label>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mt-2">
              <div>
                <div>确认回看天数</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCond3.确认回看天数} onChange={e => setACond3(p => ({...p, 确认回看天数: parseInt(e.target.value||'0',10)}))} />
              </div>
              <div>
                <div>确认均线窗口</div>
                <input type="text" className="mt-1 px-2 py-1 border rounded w-full" value={aCond3.确认均线窗口} onChange={e => setACond3(p => ({...p, 确认均线窗口: e.target.value}))} />
              </div>
              <div>
                <div>确认价格列</div>
                <select className="mt-1 px-2 py-1 border rounded w-full" value={aCond3.确认价格列} onChange={e => setACond3(p => ({...p, 确认价格列: e.target.value}))}>
                  <option value="close">收盘价</option>
                  <option value="high">最高价</option>
                  <option value="low">最低价</option>
                </select>
              </div>
            </div>
          </div>
          {/* 条件4：放量确认（子模块） */}
          <div className="border rounded p-3">
            <div className="font-medium mb-2">条件4：放量确认（模块化）</div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
              <div className="border rounded p-3">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium">子模块1：VR1 放量</div>
                  <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={aCondVol.vr1_enabled} onChange={e => setACondVol(p => ({...p, vr1_enabled: e.target.checked}))} /><span>启用</span></label>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-3">
                  <div>
                    <div>回看天数</div>
                    <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCondVol.对比天数} onChange={e => setACondVol(p => ({...p, 对比天数: parseInt(e.target.value||'10',10)}))} />
                  </div>
                  <div>
                    <div>倍数阈值</div>
                    <input type="number" step="0.01" className="mt-1 px-2 py-1 border rounded w-full" value={aCondVol.倍数} onChange={e => setACondVol(p => ({...p, 倍数: parseFloat(e.target.value||'2')}))} />
                  </div>
                </div>
              </div>
              <div className="border rounded p-3">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium">子模块2：量均比较（短&gt;长）</div>
                  <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={aCondVol.vma_cmp_enabled} onChange={e => setACondVol(p => ({...p, vma_cmp_enabled: e.target.checked}))} /><span>启用</span></label>
                </div>
                <div className="mt-2 grid grid-cols-2 gap-3">
                  <div>
                    <div>短期天数</div>
                    <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCondVol.短期天数} onChange={e => setACondVol(p => ({...p, 短期天数: parseInt(e.target.value||'5',10)}))} />
                  </div>
                  <div>
                    <div>长期天数</div>
                    <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCondVol.长期天数} onChange={e => setACondVol(p => ({...p, 长期天数: parseInt(e.target.value||'10',10)}))} />
                  </div>
                </div>
              </div>
              <div className="border rounded p-3">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium">子模块3：近X日量连升（严格递增）</div>
                  <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={aCondVol.vol_up_enabled} onChange={e => setACondVol(p => ({...p, vol_up_enabled: e.target.checked}))} /><span>启用</span></label>
                </div>
                <div className="mt-2 grid grid-cols-1 gap-3">
                  <div>
                    <div>X（日）</div>
                    <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCondVol.量连升天数} onChange={e => setACondVol(p => ({...p, 量连升天数: parseInt(e.target.value||'3',10)}))} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* B 点参数 */}
      <div className="bg-white p-4 rounded shadow mb-4">
        <div className="font-semibold mb-2">B 点参数</div>
        <div className="grid grid-cols-1 gap-3">
          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件1：时间要求</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond1.enabled} onChange={e => setBCond1(p => ({...p, enabled: e.target.checked}))} /><span>启用</span></label>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mt-2">
              <div>
                <div>从 A 点到今天至少经过（天）</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond1.min_days_from_a} onChange={e => setBCond1(p => ({...p, min_days_from_a: parseInt(e.target.value||'60',10)}))} />
              </div>
              <div>
                <div>从 A 点到今天不超过（天，可留空）</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond1.max_days_from_a as any} onChange={e => setBCond1(p => ({...p, max_days_from_a: e.target.value === '' ? '' : parseInt(e.target.value,10)}))} placeholder="留空表示不限制" />
              </div>
              <div className="flex items-center gap-2 col-span-2">
                <input type="checkbox" className="mr-2" checked={bCond1.allow_multi_b_per_a} onChange={e => setBCond1(p => ({...p, allow_multi_b_per_a: e.target.checked}))} />
                <span>是否允许一个A点可以对应多个B点</span>
              </div>
            </div>
          </div>

          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件2：短期线在长期线之上</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond2MA.enabled} onChange={e => setBCond2MA(p => ({...p, enabled: e.target.checked}))} /><span>启用</span></label>
            </div>
            <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div>
                <div>短期窗口</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond2MA.above_maN_window} onChange={e => setBCond2MA(p => ({...p, above_maN_window: parseInt(e.target.value||'5',10)}))} />
              </div>
              <div>
                <div>长期窗口</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond2MA.long_ma_days} onChange={e => setBCond2MA(p => ({...p, long_ma_days: parseInt(e.target.value||'60',10)}))} />
              </div>
              <div>
                <div>在上方天数（累计）</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond2MA.above_maN_days} onChange={e => setBCond2MA(p => ({...p, above_maN_days: parseInt(e.target.value||'15',10)}))} />
              </div>
              <div>
                <div>在上方比例%</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond2MA.above_maN_ratio as any} onChange={e => setBCond2MA(p => ({...p, above_maN_ratio: parseInt(e.target.value||'60',10)}))} />
              </div>
            </div>
          </div>

          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件3：接近长期线 + 阴线/收≤昨收</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond2.enabled} onChange={e => setBCond2(p => ({...p, enabled: e.target.checked}))} /><span>启用</span></label>
            </div>
            <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div>
                <div>比较价格</div>
                <select className="mt-1 px-2 py-1 border rounded w-full" value={bCond2.touch_price} onChange={e => setBCond2(p => ({...p, touch_price: e.target.value as any}))}>
                  <option value="low">最低价</option>
                  <option value="close">收盘价</option>
                </select>
              </div>
              <div>
                <div>比较关系</div>
                <select className="mt-1 px-2 py-1 border rounded w-full" value={bCond2.touch_relation} onChange={e => setBCond2(p => ({...p, touch_relation: e.target.value as any}))}>
                  <option value="le">≤</option>
                  <option value="lt">＜</option>
                </select>
              </div>
              <div className="flex items-center gap-2"><input type="checkbox" checked={bCond2.require_bearish} onChange={e => setBCond2(p => ({...p, require_bearish: e.target.checked}))} /><span>要求阴线</span></div>
              <div className="flex items-center gap-2"><input type="checkbox" checked={bCond2.require_close_le_prev} onChange={e => setBCond2(p => ({...p, require_close_le_prev: e.target.checked}))} /><span>收≤昨收</span></div>
            </div>
          </div>

          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件4：量能上限（VR1）</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond4VR.enabled} onChange={e => setBCond4VR(p => ({...p, enabled: e.target.checked}))} /><span>启用</span></label>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-3 text-sm">
              <div><div>VR1上限</div><input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond4VR.vr1_max as any} onChange={e => setBCond4VR(p => ({...p, vr1_max: e.target.value === '' ? '' : parseFloat(e.target.value)}))} /></div>
              <div><div>回看天数</div><input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond4VR.recent_max_vol_window} onChange={e => setBCond4VR(p => ({...p, recent_max_vol_window: parseInt(e.target.value||'10',10)}))} /></div>
            </div>
          </div>

          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件5：缩量</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond3.enabled} onChange={e => setBCond3(p => ({...p, enabled: e.target.checked}))} /><span>启用</span></label>
            </div>
            {/* 子模块1：非放量（VR1 ≤ 阈值） */}
            <div className="mt-2 border rounded p-3">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium">子模块1：非放量（VR1）</div>
                <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={(bCond3 as any).vr1_enabled ?? false} onChange={e => setBCond3(p => ({...p, vr1_enabled: e.target.checked}))} /><span>启用</span></label>
              </div>
              <div className="mt-1 text-xs text-gray-500">说明：与 C 点“放量VR1”相反，这里要求 VR1 ≤ 阈值（今天量不超过近N日最大量的若干倍）。</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div>
                  <div>参照天数 N（默认 10）</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={(bCond3 as any).recent_n ?? 10} onChange={e => setBCond3(p => ({...p, recent_n: parseInt(e.target.value||'10',10)}))} />
                </div>
                <div>
                  <div>VR1 阈值倍数（默认 1.2）</div>
                  <input type="number" step="0.01" className="mt-1 px-2 py-1 border rounded w-full" value={(bCond3 as any).vr1_max ?? 1.2} onChange={e => setBCond3(p => ({...p, vr1_max: parseFloat(e.target.value||'1.2')}))} />
                </div>
              </div>
            </div>
            {/* 子模块2：量均比较（短≤长） */}
            <div className="mt-2 border rounded p-3">
              <div className="flex items-center justify-between"><div className="text-sm font-medium">子模块2：量均比较（短≤长）</div><label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={(bCond3 as any).vma_rel_enabled ?? false} onChange={e => setBCond3(p => ({...p, vma_rel_enabled: e.target.checked}))} /><span>启用</span></label></div>
              <div className="mt-1 text-xs text-gray-500">说明：与 C 点“短量均&gt;长量均”相反，这里要求短期量均≤长期量均。</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div>
                  <div>短天数（默认 5）</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond3.short_days} onChange={e => setBCond3(p => ({...p, short_days: parseInt(e.target.value||'5',10)}))} />
                </div>
                <div>
                  <div>长天数（默认 10）</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond3.long_days} onChange={e => setBCond3(p => ({...p, long_days: parseInt(e.target.value||'10',10)}))} />
                </div>
              </div>
            </div>
            {/* 子模块3：近X日量连降（严格递减） */}
            <div className="mt-2 border rounded p-3">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium">子模块3：近X日量连降</div>
                <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={(bCond3 as any).vol_down_enabled ?? false} onChange={e => setBCond3(p => ({...p, vol_down_enabled: e.target.checked}))} /><span>启用</span></label>
              </div>
              <div className="mt-1 text-xs text-gray-500">说明：要求最近X日（含当日）成交量严格递减，体现缩量走稳。</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div>
                  <div>X（日）</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={(bCond3 as any).vol_decreasing_days ?? 3} onChange={e => setBCond3(p => ({...p, vol_decreasing_days: parseInt(e.target.value||'3',10)}))} />
                </div>
              </div>
            </div>
          </div>

          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件6：价稳</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond4.enabled} onChange={e => setBCond4(p => ({...p, enabled: e.target.checked}))} /><span>启用</span></label>
            </div>
            <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div><div>模式</div><select className="mt-1 px-2 py-1 border rounded w-full" value={bCond4.price_stable_mode} onChange={e => setBCond4(p => ({...p, price_stable_mode: e.target.value as any}))}><option value="no_new_low">不创新低</option><option value="ratio">最大跌幅</option><option value="atr">ATR</option></select></div>
              <div><div>最大跌幅</div><input type="number" step="0.001" className="mt-1 px-2 py-1 border rounded w-full" value={bCond4.max_drop_ratio} onChange={e => setBCond4(p => ({...p, max_drop_ratio: parseFloat(e.target.value||'0.03')}))} /></div>
              <div><div>ATR窗口</div><input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond4.use_atr_window} onChange={e => setBCond4(p => ({...p, use_atr_window: parseInt(e.target.value||'14',10)}))} /></div>
              <div><div>ATR缓冲</div><input type="number" step="0.01" className="mt-1 px-2 py-1 border rounded w-full" value={bCond4.atr_buffer} onChange={e => setBCond4(p => ({...p, atr_buffer: parseFloat(e.target.value||'0.5')}))} /></div>
            </div>
          </div>
        </div>
      </div>

      {/* C 点参数 */}
      <div className="bg-white p-4 rounded shadow mb-4">
        <div className="font-semibold mb-2">C 点参数</div>
        <div className="grid grid-cols-1 gap-3">
          <div className="border rounded p-3">
            <div className="flex items-center justify-between"><div className="font-medium">条件1：时间窗口</div><label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond1.enabled} onChange={e => setCCond1(p => ({...p, enabled: e.target.checked}))} /><span>启用</span></label></div>
            <div className="text-xs text-gray-500 mt-1">C点距离最近的B点不超过指定天数。</div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mt-2"><div><div>最大允许天数</div><input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond1.max_days_from_b} onChange={e => setCCond1(p => ({...p, max_days_from_b: parseInt(e.target.value||'60',10)}))} /></div></div>
          </div>
          <div className="border rounded p-3">
            <div className="flex items-center justify-between"><div className="font-medium">条件2：放量确认（模块化）</div><label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond2.enabled} onChange={e => setCCond2(p => ({...p, enabled: e.target.checked}))} /><span>启用</span></label></div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm mt-2">
              <div className="border rounded p-3"><div className="flex items-center justify-between"><div className="text-sm font-medium">VR1 放量</div><label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond2.vr1_enabled} onChange={e => setCCond2(p => ({...p, vr1_enabled: e.target.checked}))} /><span>启用</span></label></div><div className="mt-2 grid grid-cols-2 gap-3"><div><div>回看天数</div><input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond2.recent_n} onChange={e => setCCond2(p => ({...p, recent_n: parseInt(e.target.value||'10',10)}))} /></div><div><div>倍数阈值</div><input type="number" step="0.01" className="mt-1 px-2 py-1 border rounded w-full" value={cCond2.vol_multiple} onChange={e => setCCond2(p => ({...p, vol_multiple: parseFloat(e.target.value||'2')}))} /></div></div></div>
              <div className="border rounded p-3"><div className="flex items-center justify-between"><div className="text-sm font-medium">量均比较</div><label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond2.vma_cmp_enabled} onChange={e => setCCond2(p => ({...p, vma_cmp_enabled: e.target.checked}))} /><span>启用</span></label></div><div className="mt-2 grid grid-cols-2 gap-3"><div><div>短期天数</div><input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond2.vma_short_days} onChange={e => setCCond2(p => ({...p, vma_short_days: parseInt(e.target.value||'5',10)}))} /></div><div><div>长期天数</div><input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond2.vma_long_days} onChange={e => setCCond2(p => ({...p, vma_long_days: parseInt(e.target.value||'10',10)}))} /></div></div></div>
              <div className="border rounded p-3"><div className="flex items-center justify-between"><div className="text-sm font-medium">近X日量连升</div><label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond2.vol_up_enabled} onChange={e => setCCond2(p => ({...p, vol_up_enabled: e.target.checked}))} /><span>启用</span></label></div><div className="mt-2 grid grid-cols-1 gap-3"><div><div>X（日）</div><input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond2.vol_increasing_days} onChange={e => setCCond2(p => ({...p, vol_increasing_days: parseInt(e.target.value||'3',10)}))} /></div></div></div>
            </div>
          </div>
          <div className="border rounded p-3">
            <div className="flex items-center justify-between"><div className="font-medium">条件3：价格与均线</div><label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond3.enabled} onChange={e => setCCond3(p => ({...p, enabled: e.target.checked}))} /><span>启用</span></label></div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mt-2"><div><div>用于比较的价格</div><select className="mt-1 px-2 py-1 border rounded w-full" value={cCond3.price_field} onChange={e => setCCond3(p => ({...p, price_field: e.target.value as any}))}><option value="close">收盘价</option><option value="high">最高价</option><option value="low">最低价</option></select></div><div><div>均线天数</div><input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond3.ma_days} onChange={e => setCCond3(p => ({...p, ma_days: parseInt(e.target.value||'60',10)}))} /></div><div><div>比较关系</div><select className="mt-1 px-2 py-1 border rounded w-full" value={cCond3.relation} onChange={e => setCCond3(p => ({...p, relation: e.target.value as any}))}><option value="ge">≥</option><option value="gt">＞</option></select></div></div>
          </div>
        </div>
      </div>

      {/* 操作与进度 */}
      <div className="bg-white p-4 rounded shadow">
        <button
          className="px-3 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded disabled:opacity-50"
          onClick={async () => {
            try {
              const payload = {
                a_params: { '条件1_长期下跌': aCond1, '条件2_短均线上穿': aCond2, '条件3_价格确认': aCond3, '条件4_放量确认': aCondVol },
                b_params: { cond1: bCond1, cond2: bCond2MA, cond3: bCond2, cond4: { enabled: bCond4VR.enabled, vr1_max: bCond4VR.vr1_max === '' ? null : bCond4VR.vr1_max, recent_max_vol_window: bCond4VR.recent_max_vol_window }, cond5: { ...bCond3, vma_short_window: bCond3.short_days, vma_long_window: bCond3.long_days, vol_compare_long_window: bCond3.vol_compare_long_window }, cond6: bCond4 },
                c_params: { cond1: cCond1, cond2: cCond2, cond3: cCond3 },
              };
              const symbols = Array.from(new Set(symbolsText.split(/[\s,]+/).map(s => s.trim()).filter(Boolean)));
              if (symbols.length) (payload as any).symbols = symbols;
              if (market) (payload as any).market = market;
              (payload as any).max_workers = maxWorkers;
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
