import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import KLineEChart from '../components/KLineEChart';
import AParamsPanel from '../components/AParamsPanel';
import BParamsPanel from '../components/BParamsPanel';
import CParamsPanel from '../components/CParamsPanel';
import { useABCDefaults } from '../hooks/useABCDefaults';

interface KLineData {
  '日期': string;
  '开盘': number;
  '收盘': number;
  '最高': number;
  '最低': number;
  '成交量': number;
}

export default function KLineChart() {
  const { aDefaults, bDefaults, cDefaults } = useABCDefaults();
  const { symbol } = useParams<{ symbol: string }>();
  const [klineData, setKlineData] = useState<KLineData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [info, setInfo] = useState<Record<string, any> | null>(null);
  const [infoError, setInfoError] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  // A点计算
  const [aPoints, setAPoints] = useState<{ date: string; price_low: number | null; price_close: number | null }[]>([]);
  const [aCond1, setACond1] = useState<{ 启用: boolean; 长均线窗口: number; 下跌跨度: number }>(aDefaults.aCond1);
  const [aCond2, setACond2] = useState<{ 启用: boolean; 短均线集合: string; 长均线窗口: number; 上穿完备窗口: number; 必须满足的短均线: string; 全部满足: boolean }>(aDefaults.aCond2);
  const [aCond3, setACond3] = useState(aDefaults.aCond3);
  const [aCondVol, setACondVol] = useState<{ 启用: boolean; vr1_enabled: boolean; 对比天数: number; 倍数: number; vma_cmp_enabled: boolean; 短期天数: number; 长期天数: number; vol_up_enabled: boolean; 量连升天数: number }>(aDefaults.aCondVol);
  const [aTable, setATable] = useState<any[]>([]);
  const [aFilter, setAFilter] = useState<'all' | 'cond1' | 'cond2' | 'cond3' | 'a_point'>('all');
  const [computingA, setComputingA] = useState(false);

  // B点计算
  const [bPoints, setBPoints] = useState<{ date: string; price_low: number | null; price_close: number | null }[]>([]);
  // 条件1：时间要求
  const [bCond1, setBCond1] = useState<{ enabled: boolean; min_days_from_a: number; max_days_from_a: number | ''; allow_multi_b_per_a: boolean }>(bDefaults.bCond1);
  // 条件2：均线关系（短在长上）
  const [bCond2MA, setBCond2MA] = useState<{ enabled: boolean; above_maN_window: number; above_maN_days: number; above_maN_consecutive: boolean; max_maN_below_days: number; long_ma_days: number; above_maN_ratio?: number }>(bDefaults.bCond2MA);
  // 条件3：接近长期线 + 阴线/收≤昨收（无VR1）
  const [bCond2, setBCond2] = useState<{ enabled: boolean; touch_price: 'low'|'close'; touch_relation: 'le'|'lt'; require_bearish: boolean; require_close_le_prev: boolean; long_ma_days: number }>(bDefaults.bCond2);
  // 条件4：量能上限（VR1）
  const [bCond4VR, setBCond4VR] = useState<{ enabled: boolean; vr1_max: number | ''; recent_max_vol_window: number }>(bDefaults.bCond4VR);
  // 条件4：干缩
  const [bCond3, setBCond3] = useState<{ enabled: boolean; dryness_ratio_max: number; require_vol_le_vma10: boolean; dryness_recent_window: number; dryness_recent_min_days: number; short_days: number; long_days: number; vol_compare_long_window: number; vr1_enabled?: boolean; vma_rel_enabled?: boolean; vol_down_enabled?: boolean; vol_decreasing_days?: number }>(bDefaults.bCond3);
  // 条件5：价稳
  const [bCond4, setBCond4] = useState<{ enabled: boolean; price_stable_mode: 'no_new_low'|'ratio'|'atr'; max_drop_ratio: number; use_atr_window: number; atr_buffer: number }>(bDefaults.bCond4);
  const [bTable, setBTable] = useState<any[]>([]);
  const [bFilter, setBFilter] = useState<'all' | 'cond1' | 'cond2' | 'cond3' | 'cond4' | 'cond5' | 'cond6' | 'b_point'>('all');
  const [computingB, setComputingB] = useState(false);

  // C点计算
  const [cPoints, setCPoints] = useState<{ date: string; price_low: number | null; price_close: number | null }[]>([]);
  const [cTable, setCTable] = useState<any[]>([]);
  const [computingC, setComputingC] = useState(false);
  const [cFilter, setCFilter] = useState<'all' | 'cond1' | 'cond2' | 'cond3' | 'c_point'>('all');
  const [cCond1, setCCond1] = useState<{ enabled: boolean; max_days_from_b: number }>(cDefaults.cCond1);
  const [cCond2, setCCond2] = useState<{ enabled: boolean; vr1_enabled: boolean; recent_n: number; vol_multiple: number; vma_cmp_enabled: boolean; vma_short_days: number; vma_long_days: number; vol_up_enabled: boolean; vol_increasing_days: number }>(
    cDefaults.cCond2
  );
  const [cCond3, setCCond3] = useState<{ enabled: boolean; price_field: 'close'|'high'|'low'; ma_days: number; relation: 'ge'|'gt' }>(cDefaults.cCond3);

  const downloadCSV = (filename: string, headers: string[], rows: (string|number|null|boolean)[][]) => {
    const escape = (v: any) => {
      if (v == null) return '';
      const s = String(v);
      return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
    };
    const lines = [headers.map(escape).join(',')].concat(rows.map(r => r.map(escape).join(',')));
    const csv = '\uFEFF' + lines.join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(url); a.remove(); }, 0);
  };

  type AuxType = '普通连线' | '移动均线';
  type DataField = '开盘' | '收盘' | '最高' | '最低';
  interface AuxLine {
    id: string;
    type: AuxType;
    field: DataField;
    window?: number;
    name: string;
  }
  const [auxLines, setAuxLines] = useState<AuxLine[]>([
    { id: `ma-5`, type: '移动均线', field: '收盘', window: 5, name: 'MA5' },
    { id: `ma-10`, type: '移动均线', field: '收盘', window: 10, name: 'MA10' },
    { id: `ma-60`, type: '移动均线', field: '收盘', window: 60, name: 'MA60' },
  ]);
  const [lineCounters, setLineCounters] = useState<{ [k in AuxType]: number }>({ '普通连线': 0, '移动均线': 0 });
  interface AuxEditor { id: string; type: AuxType; field: DataField; window: number; name: string; addedLineId?: string; }
  const [auxEditors, setAuxEditors] = useState<AuxEditor[]>([]);

  // 交易量辅助线（仅移动均线）
  interface VolAuxLine { id: string; window: number; name: string; }
  interface VolAuxEditor { id: string; window: number; name: string; addedLineId?: string; }
  const [volAuxLines, setVolAuxLines] = useState<VolAuxLine[]>([
    { id: 'vma-5', window: 5, name: 'CMA5' },
    { id: 'vma-10', window: 10, name: 'CMA10' },
  ]);
  const [volLineCounter, setVolLineCounter] = useState(2);
  const [volAuxEditors, setVolAuxEditors] = useState<VolAuxEditor[]>([]);

  useEffect(() => {
    const fetchAll = async () => {
      if (!symbol) return;
      setLoading(true);
      setError('');
      setInfoError('');
      try {
        const [kRes, iRes] = await Promise.all([
          fetch(`/api/stocks/${symbol}`),
          fetch(`/api/stocks/${symbol}/info`),
        ]);
        if (!kRes.ok) throw new Error('Failed to fetch K-line data');
        const kd = await kRes.json();
        setKlineData(kd);

        if (iRes.ok) {
          const idata = await iRes.json();
          setInfo(idata);
        } else {
          setInfo(null);
          setInfoError('无法获取股票详情');
        }
      } catch (e: any) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    };

    fetchAll();
  }, [symbol]);

  
  if (loading) {
    return <div className="container mx-auto p-4">加载K线数据中...</div>;
  }

  if (error) {
    return <div className="container mx-auto p-4 text-red-500">{error}</div>;
  }

  return (
    <div className="container mx-auto p-4">
      {/* 股票详情信息卡片 */}
      <div className="bg-white shadow rounded-md p-4 mb-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-xl font-semibold">
              {info?.['股票简称'] || symbol} <span className="text-gray-500 text-base">({info?.['股票代码'] || symbol})</span>
            </div>
            <div className="text-sm text-gray-600 mt-1">
              <span className="mr-4">行业：{info?.['行业'] ?? '-'}</span>
              <span className="mr-4">上市时间：{info?.['上市时间'] ?? '-'}</span>
            </div>
          </div>
          <div className="text-right text-sm text-gray-700">
            <div>总市值：{info?.['总市值'] ?? '-'}</div>
            <div>流通市值：{info?.['流通市值'] ?? '-'}</div>
            <div>总股本：{info?.['总股本'] ?? '-'}</div>
            <div>流通股：{info?.['流通股'] ?? '-'}</div>
          </div>
        </div>
        {infoError && <div className="text-xs text-red-500 mt-2">{infoError}</div>}
      </div>
      <div className="mb-3">
        <button
          className="px-3 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded disabled:opacity-50"
          onClick={async () => {
            if (!symbol) return;
            setRefreshing(true);
            setError('');
            try {
              const r = await fetch(`/api/stocks/${symbol}/refresh`, { method: 'POST' });
              if (!r.ok) throw new Error('刷新失败');
              const data = await r.json();
              setKlineData(data);
            } catch (e: any) {
              setError(e.message);
            } finally {
              setRefreshing(false);
            }
          }}
          disabled={refreshing}
        >
          {refreshing ? '刷新中...' : '刷新数据'}
        </button>
      </div>
      <KLineEChart symbol={symbol || ''} klineData={klineData} aPoints={aPoints} bPoints={bPoints} cPoints={cPoints} auxLines={auxLines as any} volAuxLines={volAuxLines as any} />
      {/* 添加辅助线 */}
      <div className="mt-4 bg-white p-4 rounded shadow">
        {/* 动态编辑行列表 */}
        <div className="space-y-3">
          {auxEditors.map((ed) => (
            <div key={ed.id} className="flex flex-wrap items-end gap-4 border border-gray-200 rounded p-3">
              <div>
                <label className="block text-sm font-medium text-gray-700">辅助线类型</label>
                <select value={ed.type} onChange={e => {
                  const v = e.target.value as AuxType;
                  setAuxEditors(prev => prev.map(x => x.id === ed.id ? { ...x, type: v } : x));
                }} className="mt-1 px-2 py-1 border rounded">
                  <option value="普通连线">普通连线</option>
                  <option value="移动均线">移动均线</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">数据类型</label>
                <select value={ed.field} onChange={e => {
                  const v = e.target.value as DataField;
                  setAuxEditors(prev => prev.map(x => x.id === ed.id ? { ...x, field: v } : x));
                }} className="mt-1 px-2 py-1 border rounded">
                  <option value="开盘">开盘价</option>
                  <option value="收盘">收盘价</option>
                  <option value="最高">最高价</option>
                  <option value="最低">最低价</option>
                </select>
              </div>
              {ed.type === '移动均线' && (
                <div>
                  <label className="block text-sm font-medium text-gray-700">窗口期</label>
                  <input type="number" min={1} value={ed.window} onChange={e => {
                    const v = parseInt(e.target.value || '1', 10);
                    setAuxEditors(prev => prev.map(x => x.id === ed.id ? { ...x, window: Math.max(1, v) } : x));
                  }} className="mt-1 px-2 py-1 border rounded w-24" />
                </div>
              )}
              <div className="flex-1 min-w-[180px]">
                <label className="block text-sm font-medium text-gray-700">自定义名称（可选）</label>
                <input type="text" value={ed.name} onChange={e => {
                  const v = e.target.value;
                  setAuxEditors(prev => prev.map(x => x.id === ed.id ? { ...x, name: v } : x));
                }} placeholder="留空使用自动命名" className="mt-1 px-2 py-1 border rounded w-full" />
              </div>
              <div className="ml-auto flex items-center gap-2">
                <button
                  className="px-3 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded"
                  onClick={() => {
                    const nextIdx = (lineCounters[ed.type] ?? 0) + 1;
                    const defaultName = `${ed.type}${nextIdx}`;
                    const name = (ed.name && ed.name.trim()) ? ed.name.trim() : defaultName;
                    if (ed.addedLineId) {
                      // 更新已添加的辅助线
                      setAuxLines(prev => prev.map(l => l.id === ed.addedLineId ? {
                        ...l, type: ed.type, field: ed.field, window: ed.type === '移动均线' ? Math.max(1, ed.window || 5) : undefined, name
                      } : l));
                    } else {
                      // 新增辅助线
                      const id = `${ed.type}-${Date.now()}-${Math.random().toString(36).slice(2,8)}`;
                      const toAdd: AuxLine = { id, type: ed.type, field: ed.field, name };
                      if (ed.type === '移动均线') toAdd.window = Math.max(1, ed.window || 5);
                      setAuxLines(prev => [...prev, toAdd]);
                      setAuxEditors(prev => prev.map(x => x.id === ed.id ? { ...x, addedLineId: id, name } : x));
                      setLineCounters(prev => ({ ...prev, [ed.type]: nextIdx }));
                    }
                  }}
                >添加</button>
                <button
                  className="px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 hover:bg-gray-50 rounded"
                  onClick={() => {
                    // 删除编辑行，若已添加辅助线，同步删除该辅助线
                    setAuxEditors(prev => prev.filter(x => x.id !== ed.id));
                    if (ed.addedLineId) {
                      setAuxLines(prev => prev.filter(l => l.id !== ed.addedLineId));
                    }
                  }}
                >删除</button>
              </div>
            </div>
          ))}
        </div>

        {/* 新增一行编辑器 */}
        <div className="mt-3">
          <button
            className="px-3 py-2 text-sm font-medium text-white bg-gray-700 hover:bg-gray-800 rounded"
            onClick={() => {
              const newEd: AuxEditor = {
                id: `ed-${Date.now()}-${Math.random().toString(36).slice(2,8)}`,
                type: '普通连线',
                field: '收盘',
                window: 5,
                name: '',
              };
              setAuxEditors(prev => [...prev, newEd]);
            }}
          >添加K线图辅助线</button>
        </div>
      </div>
      {computingA && (
        <div className="fixed inset-0 bg-black/30 z-50 flex items-center justify-center">
          <div className="bg-white rounded px-6 py-4 shadow text-sm">正在计算A点，请稍候...</div>
        </div>
      )}
      {computingB && (
        <div className="fixed inset-0 bg-black/30 z-50 flex items-center justify-center">
          <div className="bg-white rounded px-6 py-4 shadow text-sm">正在计算B点，请稍候...</div>
        </div>
      )}
      {computingC && (
        <div className="fixed inset-0 bg-black/30 z-50 flex items-center justify-center">
          <div className="bg-white rounded px-6 py-4 shadow text-sm">正在计算C点，请稍候...</div>
        </div>
      )}


      {/* 添加交易量辅助线 */}
      <div className="mt-4 bg-white p-4 rounded shadow">
        {/* 动态编辑行列表 */}
        <div className="space-y-3">
          {volAuxEditors.map(ed => (
            <div key={ed.id} className="flex flex-wrap items-end gap-4 border border-gray-200 rounded p-3">
              <div>
                <div className="block text-sm font-medium text-gray-700">辅助线类型</div>
                <div className="mt-1 text-sm">移动均线</div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">数据类型</label>
                <div className="mt-1 text-sm">成交量</div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700">窗口期</label>
                <input type="number" min={1} value={ed.window} onChange={e => {
                  const v = parseInt(e.target.value || '1', 10);
                  setVolAuxEditors(prev => prev.map(x => x.id === ed.id ? { ...x, window: Math.max(1, v) } : x));
                }} className="mt-1 px-2 py-1 border rounded w-24" />
              </div>
              <div className="flex-1 min-w-[180px]">
                <label className="block text-sm font-medium text-gray-700">自定义名称（可选）</label>
                <input type="text" value={ed.name} onChange={e => setVolAuxEditors(prev => prev.map(x => x.id === ed.id ? { ...x, name: e.target.value } : x))} placeholder="留空使用自动命名" className="mt-1 px-2 py-1 border rounded w-full" />
              </div>
              <div className="ml-auto flex items-center gap-2">
                <button
                  className="px-3 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded"
                  onClick={() => {
                    const nextIdx = volLineCounter + 1;
                    const defaultName = `成交量MA${ed.window}-${nextIdx}`;
                    const name = (ed.name && ed.name.trim()) ? ed.name.trim() : defaultName;
                    if (ed.addedLineId) {
                      setVolAuxLines(prev => prev.map(l => l.id === ed.addedLineId ? { ...l, window: Math.max(1, ed.window), name } : l));
                      setVolAuxEditors(prev => prev.map(x => x.id === ed.id ? { ...x, name } : x));
                    } else {
                      const id = `vma-${Date.now()}-${Math.random().toString(36).slice(2,8)}`;
                      const toAdd: VolAuxLine = { id, window: Math.max(1, ed.window), name };
                      setVolAuxLines(prev => [...prev, toAdd]);
                      setVolAuxEditors(prev => prev.map(x => x.id === ed.id ? { ...x, addedLineId: id, name } : x));
                      setVolLineCounter(nextIdx);
                    }
                  }}
                >添加</button>
                <button
                  className="px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 hover:bg-gray-50 rounded"
                  onClick={() => {
                    setVolAuxEditors(prev => prev.filter(x => x.id !== ed.id));
                    if (ed.addedLineId) setVolAuxLines(prev => prev.filter(l => l.id !== ed.addedLineId));
                  }}
                >删除</button>
              </div>
            </div>
          ))}
        </div>

        {/* 新增一行交易量编辑器 */}
        <div className="mt-3">
          <button
            className="px-3 py-2 text-sm font-medium text-white bg-gray-700 hover:bg-gray-800 rounded"
            onClick={() => {
              const ed: VolAuxEditor = { id: `ved-${Date.now()}-${Math.random().toString(36).slice(2,8)}`, window: 5, name: '' };
              setVolAuxEditors(prev => [...prev, ed]);
            }}
          >添加交易量辅助线</button>
        </div>
      </div>

      <div className="mt-4">
        <Link to="/" className="text-indigo-600 hover:text-indigo-900">返回股票列表</Link>
      </div>

      {/* A 点参数（复用通用面板） */}
      <AParamsPanel
        aCond1={aCond1} setACond1={setACond1}
        aCond2={aCond2} setACond2={setACond2}
        aCond3={aCond3} setACond3={setACond3}
        aCondVol={aCondVol} setACondVol={setACondVol}
        computingA={computingA}
        actions={(
          <div>
            <button
              className="px-3 py-2 text-sm font-medium text-white bg-green-600 hover:bg-green-700 rounded disabled:opacity-50"
              onClick={async () => {
                if (!symbol) return;
                try {
                  setComputingA(true);
                  const body = { '条件1_长期下跌': aCond1, '条件2_短均线上穿': aCond2, '条件3_价格确认': aCond3, '条件4_放量确认': aCondVol };
                  const r = await fetch(`/api/stocks/${symbol}/a_points_v2`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
                  if (!r.ok) throw new Error('计算A点失败');
                  const data = await r.json();
                  setAPoints(data.points || []);
                  setATable(Array.isArray(data.table) ? data.table : []);
                } catch (e: any) {
                  alert(e.message || '请求失败');
                } finally {
                  setComputingA(false);
                }
              }}
              disabled={computingA}
            >计算A点</button>
          </div>
        )}
      />
      {/* A点诊断表格 */}
      {aTable.length > 0 && (() => {
        type Col = { key: string; header: string; align: 'left'|'right'|'center'; sticky?: boolean; groupStart?: boolean };
        const cols: Col[] = [
          { key: 'date', header: '日期', align: 'left', sticky: true },
          { key: 'open', header: '开盘', align: 'right' },
          { key: 'close', header: '收盘', align: 'right' },
          { key: 'low', header: '最低', align: 'right' },
          { key: 'high', header: '最高', align: 'right' },
          { key: 'volume', header: '成交量', align: 'right' },
        ];

        // 条件1分组
        if (aCond1.启用) {
          cols.push({ key: 'ma_long_t1', header: `MA${aCond1.长均线窗口}(t-1)`, align: 'right', groupStart: true });
          cols.push({ key: 'ma_long_t1_prev', header: `MA${aCond1.长均线窗口}(t-1-${aCond1.下跌跨度})`, align: 'right' });
          cols.push({ key: 'cond1', header: '满足条件1', align: 'center' });
        }

        // 条件2分组
        if (aCond2.启用) {
          const parseInts = (s: string) => Array.from(new Set((s||'').split(',').map(x => parseInt(x.trim(),10)).filter(n => Number.isFinite(n) && n>0)));
          const shorts = (aCond2.必须满足的短均线 && aCond2.必须满足的短均线.trim()) ? parseInts(aCond2.必须满足的短均线) : parseInts(aCond2.短均线集合);
          let first = true;
          for (const k of shorts) {
            cols.push({ key: `ma_${k}`, header: `MA${k}`, align: 'right', groupStart: first });
            first = false;
          }
          cols.push({ key: `ma_${aCond2.长均线窗口}`, header: `MA${aCond2.长均线窗口}`, align: 'right' });
          cols.push({ key: 'cond2', header: '满足条件2', align: 'center' });
        }

        // 条件3分组
        if (aCond3.启用) {
          cols.push({ key: 'confirm_cross_cnt', header: '确认上穿次数', align: 'right', groupStart: true });
          cols.push({ key: 'cond3', header: '满足条件3', align: 'center' });
        }

        // 条件4（放量）分组
        if (aCondVol.启用) {
          let started = false;
          if (aCondVol.vr1_enabled) {
            cols.push({ key: 'prevXmax', header: `前${aCondVol.对比天数}日量最大`, align: 'right', groupStart: !started }); started = true;
            cols.push({ key: 'vol_ratio', header: '当日/前最大', align: 'right' });
            cols.push({ key: 'VR1', header: 'VR1', align: 'right' });
            cols.push({ key: 'c4_vr1_ok', header: '放量VR1', align: 'center' });
          }
          if (aCondVol.vma_cmp_enabled) {
            cols.push({ key: 'vmaD', header: `VMA${aCondVol.短期天数}`, align: 'right', groupStart: !started }); started = true;
            cols.push({ key: 'vmaF', header: `VMA${aCondVol.长期天数}`, align: 'right' });
            cols.push({ key: 'c4_vma_ok', header: '量均比较', align: 'center' });
          }
          if (aCondVol.vol_up_enabled) {
            cols.push({ key: 'c4_up_ok', header: `近${aCondVol.量连升天数}日量连升`, align: 'center', groupStart: !started }); started = true;
          }
          cols.push({ key: 'cond4', header: '满足条件4', align: 'center' });
        }

        cols.push({ key: 'A_point', header: 'A点', align: 'center', groupStart: true });
        return (
          <div className="mt-4 bg-white p-4 rounded shadow">
            <div className="flex items-center justify-between mb-2">
              <div className="font-medium">A点诊断数据</div>
              <div className="text-sm flex items-center gap-2">
                <span>筛选:</span>
                <select className="px-2 py-1 border rounded" value={aFilter} onChange={e => setAFilter(e.target.value as any)}>
                  <option value="all">全部</option>
                  <option value="cond1">满足条件1</option>
                  <option value="cond2">满足条件2</option>
                  <option value="cond3">满足条件3</option>
                  <option value="a_point">A点</option>
                </select>
                <button
                  className="ml-2 px-2 py-1 border rounded hover:bg-gray-50"
                  onClick={() => {
                    const filtered = aTable.filter(r => {
                      if (aFilter === 'all') return true;
                      if (aFilter === 'a_point') return !!r.A_point;
                      return !!(r as any)[aFilter];
                    });
                    const headers = cols.map(c => c.header);
                    const rows = filtered.map((r: any) => cols.map(c => {
                      const v = r[c.key as string];
                      if ((c.key as string) === 'date') return r.date;
                      if (typeof v === 'number') return (+v).toFixed(2);
                      if (typeof v === 'boolean') return v ? 'TRUE' : '';
                      if (v == null) return '';
                      return v;
                    }));
                    downloadCSV(`${symbol || 'stock'}-A点诊断数据.csv`, headers, rows);
                  }}
                >导出CSV</button>
              </div>
            </div>
            <div className="overflow-auto" style={{ maxHeight: 360 }}>
              <table className="min-w-[1200px] table-fixed border-collapse">
                <thead className="sticky top-0 z-20">
                  <tr className="bg-gray-50 text-xs">
                    {cols.map((c, i) => (
                      <th
                        key={c.key}
                        className={`border px-2 py-1 text-gray-700 whitespace-nowrap ${c.align === 'right' ? 'text-right' : c.align === 'center' ? 'text-center' : 'text-left'} ${i===0 ? 'sticky left-0 z-30 bg-gray-50' : ''} ${c.groupStart ? 'border-l-4 border-gray-400' : ''}`}
                      >{c.header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {aTable
                    .filter(r => {
                      if (aFilter === 'all') return true;
                      if (aFilter === 'a_point') return !!r.A_point;
                      return !!(r as any)[aFilter];
                    })
                    .map((r, idx) => (
                      <tr key={idx} className="text-xs">
                        {cols.map((c, i) => {
                          const v = (r as any)[c.key];
                          let text: any = v;
                          if (c.key === 'date') text = r.date;
                          else if (typeof v === 'number') text = (+v).toFixed(2);
                          else if (typeof v === 'boolean') text = v ? '✓' : '';
                          else if (v == null) text = '-';
                          const alignClass = c.align === 'right' ? 'text-right' : c.align === 'center' ? 'text-center' : 'text-left';
                          const stickyClass = i===0 ? 'sticky left-0 z-10 bg-white' : '';
                          const groupClass = c.groupStart ? 'border-l-4 border-gray-400' : '';
                          return (
                            <td key={c.key} className={`border px-2 py-1 whitespace-nowrap ${alignClass} ${stickyClass} ${groupClass}`}>{text}</td>
                          );
                        })}
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })()}

      {/* B 点参数（复用通用面板） */}
      <BParamsPanel
        bCond1={bCond1} setBCond1={setBCond1}
        bCond2MA={bCond2MA} setBCond2MA={setBCond2MA}
        bCond2={bCond2} setBCond2={setBCond2}
        bCond4VR={bCond4VR} setBCond4VR={setBCond4VR}
        bCond3={bCond3} setBCond3={setBCond3}
        bCond4={bCond4} setBCond4={setBCond4}
        computingB={computingB}
        actions={(
          <div>
            <button
              className="px-3 py-2 text-sm font-medium text-white bg-emerald-600 hover:bg-emerald-700 rounded disabled:opacity-50"
              onClick={async () => {
                if (!symbol) return;
                try {
                  setComputingB(true);
                  const body: any = {
                    cond1: { enabled: bCond1.enabled, min_days_from_a: bCond1.min_days_from_a, max_days_from_a: bCond1.max_days_from_a === '' ? null : bCond1.max_days_from_a, allow_multi_b_per_a: bCond1.allow_multi_b_per_a },
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
                    a_points_dates: aPoints.map(p => p.date),
                  };
                  const r = await fetch(`/api/stocks/${symbol}/b_points`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
                  if (!r.ok) throw new Error('计算B点失败');
                  const data = await r.json();
                  setBPoints(data.points || []);
                  setBTable(Array.isArray(data.table) ? data.table : []);
                } catch (e: any) {
                  alert(e.message || '请求失败');
                } finally {
                  setComputingB(false);
                }
              }}
              disabled={computingB || aTable.length === 0}
              title={aTable.length === 0 ? '请先计算A点' : ''}
            >计算B点</button>
            {aTable.length === 0 && (
              <span className="ml-3 text-xs text-gray-500">请先计算A点后再计算B点</span>
            )}
          </div>
        )}
      />
      {/* B点诊断表格 */}
      {bTable.length > 0 && (() => {
        type Col = { key: string; header: string; align: 'left'|'right'|'center'; groupStart?: boolean };
        const cols: Col[] = [
          { key: 'date', header: '日期', align: 'left' },
          { key: 'open', header: '开盘', align: 'right' },
          { key: 'close', header: '收盘', align: 'right' },
          { key: 'low', header: '最低', align: 'right' },
          { key: 'high', header: '最高', align: 'right' },
          { key: 'volume', header: '成交量', align: 'right' },
        ];

        // 条件1
        if (bCond1.enabled) {
          cols.push({ key: 'days_since_A', header: 'A→B天数', align: 'right', groupStart: true });
          cols.push({ key: 'cond1', header: '满足条件1', align: 'center' });
        }

        // 条件2：短在长上（比例/累计/连续三种在后端统一为 cond2 & ratio_pct）
        if (bCond2MA.enabled) {
          cols.push({ key: 'cond2_ratio_pct', header: '在上方的比例%', align: 'right', groupStart: true });
          cols.push({ key: 'cond2', header: '满足条件2', align: 'center' });
        }

        // 条件3：接近长期线 + 阴线/收≤昨收
        if (bCond2.enabled) {
          cols.push({ key: 'ma_long', header: `MA${bCond2.long_ma_days}`, align: 'right', groupStart: true });
          cols.push({ key: 'bearish', header: '阴线', align: 'center' });
          cols.push({ key: 'close_le_prev', header: '收≤昨收', align: 'center' });
          cols.push({ key: 'touch_ma60', header: '触及/击长期均线', align: 'center' });
          cols.push({ key: 'cond3', header: '满足条件3', align: 'center' });
        }

        // 条件4：量能上限（VR1）
        if (bCond4VR.enabled) {
          cols.push({ key: 'vr1', header: 'VR1', align: 'right', groupStart: true });
          cols.push({ key: 'vr1_ok', header: 'VR1≤阈值', align: 'center' });
          cols.push({ key: 'cond4', header: '满足条件4', align: 'center' });
        }

        // 条件5：缩量（子模块）
        if (bCond3.enabled) {
          let started = false;
          if ((bCond3 as any).vr1_enabled) {
            cols.push({ key: 'c5_vr1_ok', header: 'VR1≤阈值', align: 'center', groupStart: !started }); started = true;
          }
          if ((bCond3 as any).vma_rel_enabled) {
            cols.push({ key: 'c5_vma_rel_ok', header: '短量均≤长量均', align: 'center', groupStart: !started }); started = true;
          }
          // 旧式或显式比例模块
          const ratioEnabled = (bCond3 as any).ratio_enabled ?? true; // 兼容旧式默认 true
          if (ratioEnabled) {
            cols.push({ key: 'dryness_ratio', header: '干缩比(短/长量均)', align: 'right', groupStart: !started }); started = true;
            cols.push({ key: 'c5_ratio_ok', header: '比例≤阈值', align: 'center' });
          }
          const volCmpEnabled = (bCond3 as any).vol_cmp_enabled ?? bCond3.require_vol_le_vma10;
          if (volCmpEnabled) {
            cols.push({ key: 'c5_vol_cmp_ok', header: '量≤长均', align: 'center', groupStart: !started }); started = true;
          }
          if ((bCond3 as any).vol_down_enabled) {
            cols.push({ key: 'c5_down_ok', header: '近X日量连降', align: 'center', groupStart: !started }); started = true;
            cols.push({ key: 'vol_down_streak', header: '连降天数', align: 'right' });
          }
          cols.push({ key: 'cond5', header: '满足条件5', align: 'center' });
        }

        // 条件6：价稳
        if (bCond4.enabled) {
          cols.push({ key: 'cond6_metric', header: '价稳指标', align: 'right', groupStart: true });
          cols.push({ key: 'cond6', header: '满足条件6', align: 'center' });
        }

        cols.push({ key: 'B_point', header: 'B点', align: 'center', groupStart: true });
        return (
          <div className="mt-4 bg-white p-4 rounded shadow">
            <div className="flex items-center justify-between mb-2">
              <div className="font-medium">B点诊断数据</div>
              <div className="text-sm flex items-center gap-2">
                <span>筛选:</span>
                <select className="px-2 py-1 border rounded" value={bFilter} onChange={e => setBFilter(e.target.value as any)}>
                  <option value="all">全部</option>
                  <option value="cond1">满足条件1</option>
                  <option value="cond2">满足条件2</option>
                  <option value="cond3">满足条件3</option>
                  <option value="cond4">满足条件4</option>
                  <option value="cond5">满足条件5</option>
                  <option value="cond6">满足条件6</option>
                  <option value="b_point">B点</option>
                </select>
                <button
                  className="ml-2 px-2 py-1 border rounded hover:bg-gray-50"
                  onClick={() => {
                    const filtered = bTable.filter((r: any) => {
                      if (bFilter === 'all') return true;
                      if (bFilter === 'b_point') return !!r.B_point;
                      return !!r[bFilter];
                    });
                    const headers = cols.map(c => c.header);
                    const rows = filtered.map((r: any) => cols.map(c => {
                      const key = c.key as string;
                      const v = r[key];
                      if (key === 'date') return r.date;
                      if (key === 'cond2_ratio_pct') return (v == null || isNaN(v)) ? '' : `${Math.round(+v)}%`;
                      if (typeof v === 'number') return (+v).toFixed(2);
                      if (typeof v === 'boolean') return v ? 'TRUE' : '';
                      if (v == null) return '';
                      return v;
                    }));
                    downloadCSV(`${symbol || 'stock'}-B点诊断数据.csv`, headers, rows);
                  }}
                >导出CSV</button>
              </div>
            </div>
            <div className="overflow-auto" style={{ maxHeight: 360 }}>
              <table className="min-w-[1200px] table-fixed border-collapse">
                <thead className="sticky top-0 z-20">
                  <tr className="bg-gray-50 text-xs">
                    {cols.map((c) => (
                      <th key={c.key as string} className={`border px-2 py-1 text-gray-700 whitespace-nowrap ${c.align === 'right' ? 'text-right' : c.align === 'center' ? 'text-center' : 'text-left'} ${c.groupStart ? 'border-l-4 border-gray-400' : ''}`}>{c.header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {bTable
                    .filter(r => {
                      if (bFilter === 'all') return true;
                      if (bFilter === 'b_point') return !!r.B_point;
                      return !!r[bFilter];
                    })
                    .map((r, idx) => (
                      <tr key={idx} className="text-xs">
                        {cols.map((c) => {
                          const v = (r as any)[c.key as string];
                          let text: any = v;
                          if ((c.key as string) === 'date') text = r.date;
                          else if ((c.key as string) === 'cond2_ratio_pct') {
                            text = (v == null || isNaN(v)) ? '-' : `${Math.round(+v)}%`;
                          }
                          else if (typeof v === 'number') text = (+v).toFixed(2);
                          else if (typeof v === 'boolean') text = v ? '✓' : '';
                          else if (v == null) text = '-';
                          return (
                            <td key={c.key as string} className={`border px-2 py-1 whitespace-nowrap ${c.align === 'right' ? 'text-right' : c.align === 'center' ? 'text-center' : 'text-left'} ${c.groupStart ? 'border-l-4 border-gray-400' : ''}`}>{text}</td>
                          );
                        })}
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })()}

      {/* C 点参数（复用通用面板） */}
      <CParamsPanel
        cCond1={cCond1} setCCond1={setCCond1}
        cCond2={cCond2} setCCond2={setCCond2}
        cCond3={cCond3} setCCond3={setCCond3}
        computingC={computingC}
        actions={(
          <div>
            <button
              className="px-3 py-2 text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 rounded disabled:opacity-50"
              onClick={async () => {
                if (!symbol) return;
                try {
                  setComputingC(true);
                  const body: any = {
                    cond1: cCond1,
                    cond2: cCond2,
                    cond3: cCond3,
                    a_points_dates: aPoints.map(p => p.date),
                    b_points_dates: bPoints.map(p => p.date),
                  };
                  const r = await fetch(`/api/stocks/${symbol}/c_points`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
                  if (!r.ok) throw new Error('计算C点失败');
                  const data = await r.json();
                  setCPoints(data.points || []);
                  setCTable(Array.isArray(data.table) ? data.table : []);
                } catch (e: any) {
                  alert(e.message || '请求失败');
                } finally {
                  setComputingC(false);
                }
              }}
              disabled={computingC || bPoints.length === 0}
              title={bPoints.length === 0 ? '请先计算B点' : ''}
            >计算C点</button>
            {bPoints.length === 0 && (
              <span className="ml-3 text-xs text-gray-500">请先计算B点后再计算C点</span>
            )}
          </div>
        )}
      />
    {/* C点诊断表格 */}
    {cTable.length > 0 && (() => {
        type Col = { key: string; header: string; align: 'left'|'right'|'center'; groupStart?: boolean };
        const cols: Col[] = [
          { key: 'date', header: '日期', align: 'left' },
          { key: 'open', header: '开盘', align: 'right' },
          { key: 'close', header: '收盘', align: 'right' },
          { key: 'low', header: '最低', align: 'right' },
          { key: 'high', header: '最高', align: 'right' },
          { key: 'volume', header: '成交量', align: 'right' },
        ];

        // 条件1
        if (cCond1.enabled) {
          cols.push({ key: 'days_since_B', header: '距B天数', align: 'right', groupStart: true });
          cols.push({ key: 'cond1', header: '满足条件1', align: 'center' });
        }

        // 条件2（放量：子模块）
        if (cCond2.enabled) {
          let started = false;
          if (cCond2.vr1_enabled) {
            cols.push({ key: 'vol_ratio', header: '当日/前N日最大', align: 'right', groupStart: !started }); started = true;
            cols.push({ key: 'c2_vr1_ok', header: '放量(倍数×前N最大)', align: 'center' });
          }
          if (cCond2.vma_cmp_enabled) {
            cols.push({ key: 'vma_short', header: `VMA${cCond2.vma_short_days}`, align: 'right', groupStart: !started }); started = true;
            cols.push({ key: 'vma_long', header: `VMA${cCond2.vma_long_days}`, align: 'right' });
            cols.push({ key: 'c2_vma_ok', header: '短量均>长量均', align: 'center' });
          }
          if (cCond2.vol_up_enabled) {
            cols.push({ key: 'c2_up_ok', header: `近${cCond2.vol_increasing_days}日量连升`, align: 'center', groupStart: !started }); started = true;
          }
          cols.push({ key: 'cond2', header: '满足条件2', align: 'center' });
        }

        // 条件3
        if (cCond3.enabled) {
          cols.push({ key: 'ma_Y', header: `MA(${cCond3.ma_days})`, align: 'right', groupStart: true });
          cols.push({ key: 'cond3', header: '满足条件3', align: 'center' });
        }

        cols.push({ key: 'C_point', header: 'C点', align: 'center', groupStart: true });
        return (
          <div className="mt-4 bg-white p-4 rounded shadow">
            <div className="flex items-center justify-between mb-2">
              <div className="font-medium">C点诊断数据</div>
              <div className="text-sm flex items-center gap-2">
                <span>筛选:</span>
                <select className="px-2 py-1 border rounded" value={cFilter} onChange={e => setCFilter(e.target.value as any)}>
                  <option value="all">全部</option>
                  <option value="cond1">满足条件1</option>
                  <option value="cond2">满足条件2</option>
                  <option value="cond3">满足条件3</option>
                  <option value="c_point">C点</option>
                </select>
                <button
                  className="ml-2 px-2 py-1 border rounded hover:bg-gray-50"
                  onClick={() => {
                    const filtered = cTable.filter((r: any) => {
                      if (cFilter === 'all') return true;
                      if (cFilter === 'c_point') return !!r.C_point;
                      return !!r[cFilter];
                    });
                    const headers = cols.map(c => c.header);
                    const rows = filtered.map((r: any) => cols.map(c => {
                      const key = c.key as string;
                      const v = r[key];
                      if (key === 'date') return r.date;
                      if (typeof v === 'number') return (+v).toFixed(2);
                      if (typeof v === 'boolean') return v ? 'TRUE' : '';
                      if (v == null) return '';
                      return v;
                    }));
                    downloadCSV(`${symbol || 'stock'}-C点诊断数据.csv`, headers, rows);
                  }}
                >导出CSV</button>
              </div>
            </div>
            <div className="overflow-auto" style={{ maxHeight: 360 }}>
              <table className="min-w-[1200px] table-fixed border-collapse">
                <thead className="sticky top-0 z-20">
                  <tr className="bg-gray-50 text-xs">
                    {cols.map((c) => (
                      <th key={c.key as string} className={`border px-2 py-1 text-gray-700 whitespace-nowrap ${c.align === 'right' ? 'text-right' : c.align === 'center' ? 'text-center' : 'text-left'} ${c.groupStart ? 'border-l-4 border-gray-400' : ''}`}>{c.header}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {cTable
                    .filter(r => {
                      if (cFilter === 'all') return true;
                      if (cFilter === 'c_point') return !!r.C_point;
                      return !!r[cFilter];
                    })
                    .map((r, idx) => (
                      <tr key={idx} className="text-xs">
                        {cols.map((c) => {
                          const v = (r as any)[c.key as string];
                          let text: any = v;
                          if ((c.key as string) === 'date') text = r.date;
                          else if (typeof v === 'number') text = (+v).toFixed(2);
                          else if (typeof v === 'boolean') text = v ? '✓' : '';
                          else if (v == null) text = '-';
                          return (
                            <td key={c.key as string} className={`border px-2 py-1 whitespace-nowrap ${c.align === 'right' ? 'text-right' : c.align === 'center' ? 'text-center' : 'text-left'} ${c.groupStart ? 'border-l-4 border-gray-400' : ''}`}>{text}</td>
                          );
                        })}
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
