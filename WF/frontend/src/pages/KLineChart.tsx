import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';

interface KLineData {
  '日期': string;
  '开盘': number;
  '收盘': number;
  '最高': number;
  '最低': number;
  '成交量': number;
}

export default function KLineChart() {
  const { symbol } = useParams<{ symbol: string }>();
  const [klineData, setKlineData] = useState<KLineData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [info, setInfo] = useState<Record<string, any> | null>(null);
  const [infoError, setInfoError] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  // A点计算
  const [aPoints, setAPoints] = useState<{ date: string; price_low: number | null; price_close: number | null }[]>([]);
  const [aCond1, setACond1] = useState<{ 启用: boolean; 长均线窗口: number; 下跌跨度: number }>({ 启用: true, 长均线窗口: 60, 下跌跨度: 30 });
  const [aCond2, setACond2] = useState<{ 启用: boolean; 短均线集合: string; 长均线窗口: number; 上穿完备窗口: number; 必须满足的短均线: string; 全部满足: boolean }>({ 启用: true, 短均线集合: '5,10', 长均线窗口: 60, 上穿完备窗口: 3, 必须满足的短均线: '', 全部满足: false });
  const [aCond3, setACond3] = useState<{ 启用: boolean; 确认回看天数: number; 确认均线窗口: string; 确认价格列: string }>({ 启用: false, 确认回看天数: 0, 确认均线窗口: '', 确认价格列: 'high' });
  const [aCondVol, setACondVol] = useState<{ 启用: boolean; vr1_enabled: boolean; 对比天数: number; 倍数: number; vma_cmp_enabled: boolean; 短期天数: number; 长期天数: number; vol_up_enabled: boolean; 量连升天数: number }>(
    { 启用: true, vr1_enabled: false, 对比天数: 10, 倍数: 2.0, vma_cmp_enabled: true, 短期天数: 5, 长期天数: 10, vol_up_enabled: true, 量连升天数: 3 }
  );
  const [aTable, setATable] = useState<any[]>([]);
  const [aFilter, setAFilter] = useState<'all' | 'cond1' | 'cond2' | 'cond3' | 'a_point'>('all');
  const [computingA, setComputingA] = useState(false);

  // B点计算
  const [bPoints, setBPoints] = useState<{ date: string; price_low: number | null; price_close: number | null }[]>([]);
  // 条件1：时间要求
  const [bCond1, setBCond1] = useState<{ enabled: boolean; min_days_from_a: number; max_days_from_a: number | ''; allow_multi_b_per_a: boolean }>(
    { enabled: true, min_days_from_a: 60, max_days_from_a: '', allow_multi_b_per_a: true }
  );
  // 条件2：均线关系（短在长上）
  const [bCond2MA, setBCond2MA] = useState<{ enabled: boolean; above_maN_window: number; above_maN_days: number; above_maN_consecutive: boolean; max_maN_below_days: number; long_ma_days: number; above_maN_ratio?: number }>(
    { enabled: true, above_maN_window: 5, above_maN_days: 15, above_maN_consecutive: false, max_maN_below_days: 5, long_ma_days: 60, above_maN_ratio: 60 }
  );
  // 条件3：接近长期线 + 阴线/收≤昨收（无VR1）
  const [bCond2, setBCond2] = useState<{ enabled: boolean; touch_price: 'low'|'close'; touch_relation: 'le'|'lt'; require_bearish: boolean; require_close_le_prev: boolean; long_ma_days: number }>(
    { enabled: true, touch_price: 'low', touch_relation: 'le', require_bearish: false, require_close_le_prev: false, long_ma_days: 60 }
  );
  // 条件4：量能上限（VR1）
  const [bCond4VR, setBCond4VR] = useState<{ enabled: boolean; vr1_max: number | ''; recent_max_vol_window: number }>(
    { enabled: false, vr1_max: '', recent_max_vol_window: 10 }
  );
  // 条件4：干缩
  const [bCond3, setBCond3] = useState<{ enabled: boolean; dryness_ratio_max: number; require_vol_le_vma10: boolean; dryness_recent_window: number; dryness_recent_min_days: number; short_days: number; long_days: number; vol_compare_long_window: number; vr1_enabled?: boolean; vma_rel_enabled?: boolean; vol_down_enabled?: boolean; vol_decreasing_days?: number }>(
    { enabled: true, dryness_ratio_max: 0.8, require_vol_le_vma10: true, dryness_recent_window: 0, dryness_recent_min_days: 0, short_days: 5, long_days: 10, vol_compare_long_window: 10, vr1_enabled: false, vma_rel_enabled: true, vol_down_enabled: true, vol_decreasing_days: 3 }
  );
  // 条件5：价稳
  const [bCond4, setBCond4] = useState<{ enabled: boolean; price_stable_mode: 'no_new_low'|'ratio'|'atr'; max_drop_ratio: number; use_atr_window: number; atr_buffer: number }>(
    { enabled: false, price_stable_mode: 'no_new_low', max_drop_ratio: 0.03, use_atr_window: 14, atr_buffer: 0.5 }
  );
  const [bTable, setBTable] = useState<any[]>([]);
  const [bFilter, setBFilter] = useState<'all' | 'cond1' | 'cond2' | 'cond3' | 'cond4' | 'cond5' | 'cond6' | 'b_point'>('all');
  const [computingB, setComputingB] = useState(false);

  // C点计算
  const [cPoints, setCPoints] = useState<{ date: string; price_low: number | null; price_close: number | null }[]>([]);
  const [cTable, setCTable] = useState<any[]>([]);
  const [computingC, setComputingC] = useState(false);
  const [cFilter, setCFilter] = useState<'all' | 'cond1' | 'cond2' | 'cond3' | 'c_point'>('all');
  const [cCond1, setCCond1] = useState<{ enabled: boolean; max_days_from_b: number }>({ enabled: true, max_days_from_b: 60 });
  const [cCond2, setCCond2] = useState<{ enabled: boolean; vr1_enabled: boolean; recent_n: number; vol_multiple: number; vma_cmp_enabled: boolean; vma_short_days: number; vma_long_days: number; vol_up_enabled: boolean; vol_increasing_days: number }>(
    { enabled: true, vr1_enabled: false, recent_n: 10, vol_multiple: 2.0, vma_cmp_enabled: false, vma_short_days: 5, vma_long_days: 10, vol_up_enabled: true, vol_increasing_days: 3 }
  );
  const [cCond3, setCCond3] = useState<{ enabled: boolean; price_field: 'close'|'high'|'low'; ma_days: number; relation: 'ge'|'gt' }>({ enabled: true, price_field: 'close', ma_days: 60, relation: 'ge' });

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

  const getOption = () => {
    if (!klineData.length) {
      return {};
    }

    const dates = klineData.map(item => item['日期']);
    const num = (v: any): number | null => (v == null || v === '' || isNaN(Number(v)) ? null : Number(v));
    const data = klineData.map(item => {
      let o = num((item as any)['开盘']);
      let c = num((item as any)['收盘']);
      let lo = num((item as any)['最低']);
      let hi = num((item as any)['最高']);
      if (lo == null && (o != null || c != null)) lo = Math.min(o ?? Number.POSITIVE_INFINITY, c ?? Number.POSITIVE_INFINITY);
      if (hi == null && (o != null || c != null)) hi = Math.max(o ?? Number.NEGATIVE_INFINITY, c ?? Number.NEGATIVE_INFINITY);
      if (lo != null && hi != null && lo > hi) { const t = lo; lo = hi; hi = t; }
      return [o, c, lo, hi];
    });
    const volumes = klineData.map((item, index) => [
        index,
        item['成交量'],
        item['开盘'] > item['收盘'] ? -1 : 1,
    ]);

    const legendData: string[] = [];
    const extraSeries: any[] = [];

    const numericFromField = (field: DataField) => klineData.map(d => {
      const v: any = (d as any)[field];
      return v == null || v === '' || isNaN(Number(v)) ? null : Number(v);
    });
    const ma = (arr: number[], win: number) => {
      const out: (number | null)[] = new Array(arr.length).fill(null);
      if (win <= 1) return arr.slice();
      let sum = 0;
      for (let i = 0; i < arr.length; i++) {
        const v = arr[i];
        sum += v;
        if (i >= win) sum -= arr[i - win];
        if (i >= win - 1) out[i] = +(sum / win);
      }
      return out;
    };

    for (const line of auxLines) {
      legendData.push(line.name);
      let y: (number | null)[] | number[] = [];
      if (line.type === '普通连线') {
        y = numericFromField(line.field);
      } else if (line.type === '移动均线') {
        const src = numericFromField(line.field) as number[];
        const w = Math.max(1, line.window || 5);
        y = ma(src, w);
      }
      extraSeries.push({
        name: line.name,
        type: 'line',
        data: y,
        yAxisIndex: 0,
        xAxisIndex: 0,
        showSymbol: false,
        connectNulls: true,
        smooth: false,
        lineStyle: { width: 1.5 },
        emphasis: { focus: 'series' },
      });
    }

    // 交易量辅助线（移动均线）叠加到副图
    const volArr: number[] = klineData.map(item => {
      const v: any = (item as any)['成交量'];
      return v == null || v === '' || isNaN(Number(v)) ? 0 : Number(v);
    });
    const maVol = (arr: number[], win: number) => {
      const out: (number | null)[] = new Array(arr.length).fill(null);
      if (win <= 1) return arr.slice();
      let sum = 0;
      for (let i = 0; i < arr.length; i++) {
        sum += arr[i];
        if (i >= win) sum -= arr[i - win];
        if (i >= win - 1) out[i] = +(sum / win);
      }
      return out;
    };
    for (const vline of volAuxLines) {
      legendData.push(vline.name);
      const w = Math.max(1, vline.window || 5);
      const y = maVol(volArr, w);
      extraSeries.push({
        name: vline.name,
        type: 'line',
        data: y,
        xAxisIndex: 1,
        yAxisIndex: 1,
        showSymbol: false,
        connectNulls: true,
        smooth: false,
        lineStyle: { width: 1.2, type: 'solid' },
        emphasis: { focus: 'series' },
      });
    }

    // A点标注
    if (aPoints.length > 0) {
      legendData.push('A点');
      // 构建 scatter 点
      const map = new Map<string, number>();
      dates.forEach((d, i) => map.set(d, i));
      const a_x: string[] = [];
      const a_y: number[] = [];
      for (const p of aPoints) {
        const di = map.get(p.date);
        if (di !== undefined) {
          const base = typeof p.price_low === 'number' ? p.price_low : (typeof p.price_close === 'number' ? p.price_close! : null);
          if (base !== null && !Number.isNaN(base)) {
            a_x.push(p.date);
            a_y.push(+(+base * 0.995).toFixed(2));
          }
        }
      }
      extraSeries.push({
        name: 'A点',
        type: 'scatter',
        data: a_x.map((d, i) => [d, a_y[i]]),
        xAxisIndex: 0,
        yAxisIndex: 0,
        symbol: 'triangle',
        symbolSize: 12,
        itemStyle: { color: '#2563eb' },
        tooltip: { valueFormatter: (v: any) => (v == null ? '-' : (+v).toFixed(2)) },
      });
      // 使用 [x,y] 对应到类目轴
    }

    // B点标注
    if (bPoints.length > 0) {
      legendData.push('B点');
      const map = new Map<string, number>();
      dates.forEach((d, i) => map.set(d, i));
      const bx: string[] = [];
      const by: number[] = [];
      for (const p of bPoints) {
        const di = map.get(p.date);
        if (di !== undefined) {
          const base = typeof p.price_low === 'number' ? p.price_low : (typeof p.price_close === 'number' ? p.price_close! : null);
          if (base !== null && !Number.isNaN(base)) {
            bx.push(p.date);
            by.push(+(+base * 1.005).toFixed(2));
          }
        }
      }
      extraSeries.push({
        name: 'B点',
        type: 'scatter',
        data: bx.map((d, i) => [d, by[i]]),
        xAxisIndex: 0,
        yAxisIndex: 0,
        symbol: 'diamond',
        symbolSize: 12,
        itemStyle: { color: '#10b981' },
        tooltip: { valueFormatter: (v: any) => (v == null ? '-' : (+v).toFixed(2)) },
      });
    }

    // C点标注
    if (cPoints.length > 0) {
      legendData.push('C点');
      const map = new Map<string, number>();
      dates.forEach((d, i) => map.set(d, i));
      const cx: string[] = [];
      const cy: number[] = [];
      for (const p of cPoints) {
        const di = map.get(p.date);
        if (di !== undefined) {
          const ref = typeof p.price_close === 'number' ? p.price_close! : (typeof p.price_low === 'number' ? p.price_low! : null);
          if (ref !== null && !Number.isNaN(ref)) {
            cx.push(p.date);
            cy.push(+(+ref * 1.01).toFixed(2));
          }
        }
      }
      extraSeries.push({
        name: 'C点',
        type: 'scatter',
        data: cx.map((d, i) => [d, cy[i]]),
        xAxisIndex: 0,
        yAxisIndex: 0,
        symbol: 'circle',
        symbolSize: 11,
        itemStyle: { color: '#f59e0b' },
        tooltip: { valueFormatter: (v: any) => (v == null ? '-' : (+v).toFixed(2)) },
      });
    }

    return {
      title: {
        text: `${symbol} K线图`,
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'cross' },
        formatter: (params: any[]) => {
          if (!params || !params.length) return '';
          const date = params[0].axisValueLabel || params[0].axisValue;
          const lines: string[] = [String(date)];
          // 统一从原始 klineData 取值，避免 ECharts 内部重排导致索引错位
          const k = params.find(p => p.seriesType === 'candlestick');
          if (k && typeof k.dataIndex === 'number') {
            const idx = k.dataIndex as number;
            const rec = klineData[idx] as any;
            const toNum = (v: any): number | null => (v == null || v === '' || isNaN(Number(v)) ? null : Number(v));
            let o = toNum(rec['开盘']);
            let c = toNum(rec['收盘']);
            let l = toNum(rec['最低']);
            let h = toNum(rec['最高']);
            if (l == null && (o != null || c != null)) l = Math.min(o ?? Number.POSITIVE_INFINITY, c ?? Number.POSITIVE_INFINITY);
            if (h == null && (o != null || c != null)) h = Math.max(o ?? Number.NEGATIVE_INFINITY, c ?? Number.NEGATIVE_INFINITY);
            if (l != null && h != null && l > h) { const t = l; l = h; h = t; }
            lines.push(`开:${o == null ? '-' : (+o).toFixed(2)} 收:${c == null ? '-' : (+c).toFixed(2)} 低:${l == null ? '-' : (+l).toFixed(2)} 高:${h == null ? '-' : (+h).toFixed(2)}`);
          }
          // 其他线/点
          params.forEach(p => {
            if (p.seriesType === 'line') {
              const v = p.data;
              lines.push(`${p.seriesName}: ${v == null ? '-' : (+v).toFixed(2)}`);
            } else if (p.seriesType === 'bar') {
              const v = p.data;
              lines.push(`${p.seriesName}: ${v == null ? '-' : (+v).toFixed(2)}`);
            } else if (p.seriesType === 'scatter' && p.seriesName === 'A点') {
              const v = p.data && p.data[1];
              lines.push(`${p.seriesName}: ${v == null ? '-' : (+v).toFixed(2)}`);
            } else if (p.seriesType === 'scatter' && p.seriesName === 'B点') {
              const v = p.data && p.data[1];
              lines.push(`${p.seriesName}: ${v == null ? '-' : (+v).toFixed(2)}`);
            } else if (p.seriesType === 'scatter' && p.seriesName === 'C点') {
              const v = p.data && p.data[1];
              lines.push(`${p.seriesName}: ${v == null ? '-' : (+v).toFixed(2)}`);
            }
          });
          return lines.join('<br/>');
        }
      },
      legend: { show: auxLines.length > 0 || aPoints.length > 0 || bPoints.length > 0 || cPoints.length > 0, data: legendData, top: 30 },
      grid: [
        {
          left: '10%',
          right: '8%',
          height: '50%',
        },
        {
          left: '10%',
          right: '8%',
          top: '65%',
          height: '16%',
        },
      ],
      xAxis: [
        {
          type: 'category',
          data: dates,
          scale: true,
          boundaryGap: false,
          axisLine: { onZero: false },
          splitLine: { show: false },
          min: 'dataMin',
          max: 'dataMax',
        },
        {
            type: 'category',
            gridIndex: 1,
            data: dates,
            scale: true,
            boundaryGap: false,
            axisLine: { onZero: false },
            axisTick: { show: false },
            splitLine: { show: false },
            axisLabel: { show: false },
            min: 'dataMin',
            max: 'dataMax',
        },
      ],
      yAxis: [
        {
          scale: true,
          splitArea: {
            show: true,
          },
          axisLabel: { formatter: (v: number) => (v == null ? '' : (+v).toFixed(2)) },
        },
        {
            scale: true,
            gridIndex: 1,
            splitNumber: 2,
            axisLabel: { show: true, formatter: (v: number) => (v == null ? '' : (+v).toFixed(2)) },
            axisLine: { show: false },
            axisTick: { show: false },
            splitLine: { show: false },
        },
      ],
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: [0, 1],
          start: 80,
          end: 100,
        },
        {
          show: true,
          xAxisIndex: [0, 1],
          type: 'slider',
          top: '85%',
          start: 80,
          end: 100,
        },
      ],
      series: [
        {
          name: 'K线',
          type: 'candlestick',
          data: data,
          itemStyle: {
            color: '#ec0000',
            color0: '#00da3c',
            borderColor: '#8A0000',
            borderColor0: '#008F28',
          },
        },
        {
            name: '成交量',
            type: 'bar',
            xAxisIndex: 1,
            yAxisIndex: 1,
            data: volumes.map(item => item[1]),
            itemStyle: {
                color: ({ dataIndex }: { value: number, dataIndex: number }) => {
                    const o = Number(klineData[dataIndex]['开盘']);
                    const c = Number(klineData[dataIndex]['收盘']);
                    return o > c ? '#00da3c' : '#ec0000';
                }
            },
            tooltip: { valueFormatter: (v: any) => (v == null ? '-' : (+v).toFixed(2)) }
        },
        ...extraSeries,
      ],
    };
  };

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
      <ReactECharts option={getOption()} style={{ height: '600px' }} />
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

      {/* 寻找A点（始终展示） */}
      <div className="mt-4 bg-white p-4 rounded shadow">
        <div className="font-semibold mb-2">寻找A点（基于“长期下跌 + 当日短均线上穿 + 可选价格确认”的原始逻辑）</div>
        <div className="mt-3 space-y-4">
          {/* 条件1：长期下跌 */}
          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件1：长期下跌</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={aCond1.启用} onChange={e => setACond1(p => ({...p, 启用: e.target.checked}))} disabled={computingA} /><span>启用</span></label>
            </div>
            <div className="mt-1 text-xs text-gray-500">
              说明：我们用“较长天数的收盘价平均值”画出一条“长期走势线”。
              如果这条长期走势线在“昨天”的位置比“更早一段时间”（下跌跨度天）还要低，
              就可以认为这段时间整体在往下走，属于“之前长期下跌”的背景。
            </div>
            <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div>
                <div>长均线窗口</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCond1.长均线窗口} onChange={e => setACond1(p => ({...p, 长均线窗口: parseInt(e.target.value||'60',10)}))} disabled={computingA} />
              </div>
              <div>
                <div>下跌跨度</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCond1.下跌跨度} onChange={e => setACond1(p => ({...p, 下跌跨度: parseInt(e.target.value||'30',10)}))} disabled={computingA} />
              </div>
            </div>
          </div>

          {/* 条件2：短均线上穿长均线 */}
          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件2：短均线上穿长均线</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={aCond2.启用} onChange={e => setACond2(p => ({...p, 启用: e.target.checked}))} disabled={computingA} /><span>启用</span></label>
            </div>
            <div className="mt-1 text-xs text-gray-500">
              说明：把“短期走势线”（例如5天平均）和“长期走势线”（例如60天平均）放在一起看。
              当某一天，短期线从下方越过长期线并站到长期线上方，表示短期开始转强。
              勾选“要求集合内全部满足”后，需要你设定的每条短期线，都在最近“上穿完备窗口”的天数里出现过一次这样的“从下到上”的越过，
              且在这一天它们都位于长期线上方，信号更扎实。
            </div>
            <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div>
                <div>短均线集合（逗号分隔）</div>
                <input className="mt-1 px-2 py-1 border rounded w-full" value={aCond2.短均线集合} onChange={e => setACond2(p => ({...p, 短均线集合: e.target.value}))} disabled={computingA} />
              </div>
              <div>
                <div>长均线窗口</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCond2.长均线窗口} onChange={e => setACond2(p => ({...p, 长均线窗口: parseInt(e.target.value||'60',10)}))} disabled={computingA} />
              </div>
              <div>
                <div>上穿完备窗口</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCond2.上穿完备窗口} onChange={e => setACond2(p => ({...p, 上穿完备窗口: parseInt(e.target.value||'3',10)}))} disabled={computingA} />
              </div>
              <div>
                <div>必须满足的短均线（逗号，可留空）</div>
                <input className="mt-1 px-2 py-1 border rounded w-full" value={aCond2.必须满足的短均线} onChange={e => setACond2(p => ({...p, 必须满足的短均线: e.target.value}))} disabled={computingA} />
              </div>
              <div className="col-span-2">
                <label className="inline-flex items-center space-x-2"><input type="checkbox" checked={aCond2.全部满足} onChange={e => setACond2(p => ({...p, 全部满足: e.target.checked}))} disabled={computingA} /><span>要求集合内全部满足（默认开启）</span></label>
                <div className="text-xs text-gray-500 mt-1">
                  说明：
                  - 开启：你设定的每条短期线，都需要在“上穿完备窗口”的天数里至少出现过一次“从下到上越过长期线”，且在今天这些短期线都位于长期线上方，信号更扎实但数量更少。
                  - 关闭：只要今天“任意一条”短期线出现越过并站上长期线即可，信号更多但可能更宽松。
                </div>
              </div>
            </div>
          </div>

          {/* 条件3：价格上穿确认（可选） */}
          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件3：价格上穿确认</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={aCond3.启用} onChange={e => setACond3(p => ({...p, 启用: e.target.checked}))} disabled={computingA} /><span>启用</span></label>
            </div>
            <div className="mt-1 text-xs text-gray-500">
              说明：为了进一步确认，我们会在 A 点出现之前的“确认回看天数”里再检查一次。
              价格（可以选择“最高价”或“收盘价”）至少有一次从下方越过你设定的“确认均线”，
              这有助于减少偶然波动带来的误判。
            </div>
            <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div>
                <div>确认回看天数</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCond3.确认回看天数} onChange={e => setACond3(p => ({...p, 确认回看天数: parseInt(e.target.value||'0',10)}))} disabled={computingA} />
              </div>
              <div>
                <div>确认均线窗口（留空=关闭）</div>
                <input className="mt-1 px-2 py-1 border rounded w-full" value={aCond3.确认均线窗口} onChange={e => setACond3(p => ({...p, 确认均线窗口: e.target.value}))} disabled={computingA} />
              </div>
              <div>
                <div>确认价格列</div>
                <select className="mt-1 px-2 py-1 border rounded w-full" value={aCond3.确认价格列} onChange={e => setACond3(p => ({...p, 确认价格列: e.target.value}))} disabled={computingA}>
                  <option value="high">high(最高)</option>
                  <option value="close">close(收盘)</option>
                </select>
              </div>
            </div>
          </div>

          {/* 条件4：放量确认（模块化，与C点条件2一致） */}
          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件4：放量确认</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={aCondVol.启用} onChange={e => setACondVol(p => ({...p, 启用: e.target.checked}))} disabled={computingA} /><span>启用</span></label>
            </div>
            <div className="mt-2 grid grid-cols-1 gap-3 text-sm">
              {/* 子模块1：VR1 放量 */}
              <div className="border rounded p-3">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium">子模块1：VR1 放量</div>
                  <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={aCondVol.vr1_enabled} onChange={e => setACondVol(p => ({...p, vr1_enabled: e.target.checked}))} disabled={computingA || !aCondVol.启用} /><span>启用</span></label>
                </div>
                <div className="mt-1 text-xs text-gray-500">VR1 = 今天成交量 ÷ 近N日最大成交量（不含今天），要求 VR1 ≥ 倍数。</div>
                <div className="mt-1 text-[11px] text-gray-500 leading-5">
                  示例：N=10，近10日最大量=1.00亿，今天=2.10亿 → VR1=2.10；倍数=2.0 通过，=2.2 不通过。
                </div>
                <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                  <div>
                    <div>对比天数</div>
                    <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCondVol.对比天数} onChange={e => setACondVol(p => ({...p, 对比天数: parseInt(e.target.value||'10',10)}))} disabled={computingA || !aCondVol.启用 || !aCondVol.vr1_enabled} />
                  </div>
                  <div>
                    <div>倍数</div>
                    <input type="number" step="0.1" className="mt-1 px-2 py-1 border rounded w-full" value={aCondVol.倍数} onChange={e => setACondVol(p => ({...p, 倍数: parseFloat(e.target.value||'2')}))} disabled={computingA || !aCondVol.启用 || !aCondVol.vr1_enabled} />
                  </div>
                </div>
              </div>
              {/* 子模块2：量均比较 */}
              <div className="border rounded p-3">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium">子模块2：量均比较（短&gt;长）</div>
                  <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={aCondVol.vma_cmp_enabled} onChange={e => setACondVol(p => ({...p, vma_cmp_enabled: e.target.checked}))} disabled={computingA || !aCondVol.启用} /><span>启用</span></label>
                </div>
                <div className="mt-1 text-xs text-gray-500">比较“短期成交量均线”和“长期成交量均线”，要求短期&gt;长期。</div>
                <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                  <div>
                    <div>短期天数</div>
                    <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCondVol.短期天数} onChange={e => setACondVol(p => ({...p, 短期天数: parseInt(e.target.value||'5',10)}))} disabled={computingA || !aCondVol.启用 || !aCondVol.vma_cmp_enabled} />
                  </div>
                  <div>
                    <div>长期天数</div>
                    <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCondVol.长期天数} onChange={e => setACondVol(p => ({...p, 长期天数: parseInt(e.target.value||'10',10)}))} disabled={computingA || !aCondVol.启用 || !aCondVol.vma_cmp_enabled} />
                  </div>
                </div>
              </div>
              {/* 子模块3：近X日量连升 */}
              <div className="border rounded p-3">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium">子模块3：近X日量连升（严格递增）</div>
                  <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={aCondVol.vol_up_enabled} onChange={e => setACondVol(p => ({...p, vol_up_enabled: e.target.checked}))} disabled={computingA || !aCondVol.启用} /><span>启用</span></label>
                </div>
                <div className="mt-1 text-xs text-gray-500">要求最近X日（含当日）成交量严格递增，即相邻差分均＞0。</div>
                <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                  <div>
                    <div>X（日）</div>
                    <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCondVol.量连升天数} onChange={e => setACondVol(p => ({...p, 量连升天数: parseInt(e.target.value||'3',10)}))} disabled={computingA || !aCondVol.启用 || !aCondVol.vol_up_enabled} />
                  </div>
                </div>
              </div>
            </div>
          </div>

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
        </div>
      </div>

      
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

      {/* 寻找B点（与A点同样每行一个条件，按钮在下） */}
      <div className="mt-4 bg-white p-4 rounded shadow">
        <div className="font-semibold mb-2">寻找B点（基于 A→B 条件组合）</div>
        <div className="mt-3 space-y-4">
          {/* 条件1：时间要求 */}
          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件1：时间要求</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond1.enabled} onChange={e => setBCond1(p => ({...p, enabled: e.target.checked}))} disabled={computingB} /><span>启用</span></label>
            </div>
            <div className="mt-1 text-xs text-gray-500">
              说明：从 A 点出现到今天，至少要间隔一定天数；可选上限天数。并可设置是否允许一个 A 点对应多个 B 点（默认允许）。
            </div>
            <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div>
                <div>从 A 点到今天至少经过（天）</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond1.min_days_from_a} onChange={e => setBCond1(p => ({...p, min_days_from_a: parseInt(e.target.value||'60',10)}))} disabled={computingB} />
              </div>
              <div>
                <div>从 A 点到今天不超过（天，可留空）</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond1.max_days_from_a as any}
                       onChange={e => setBCond1(p => ({...p, max_days_from_a: e.target.value === '' ? '' : parseInt(e.target.value,10)}))}
                       disabled={computingB} placeholder="留空表示不限制" />
              </div>
              <div className="flex items-center gap-2 col-span-2">
                <input type="checkbox" className="mr-2" checked={bCond1.allow_multi_b_per_a} onChange={e => setBCond1(p => ({...p, allow_multi_b_per_a: e.target.checked}))} disabled={computingB} />
                <span>是否允许一个A点可以对应多个B点</span>
              </div>
            </div>
          </div>

          {/* 条件2：短期线在长期线之上 */}
          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件2：短期线在长期线之上</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond2MA.enabled} onChange={e => setBCond2MA(p => ({...p, enabled: e.target.checked}))} disabled={computingB} /><span>启用</span></label>
            </div>
            <div className="mt-1 text-xs text-gray-500">说明：要求“短期平均线”（如 MA5）在“长期平均线”（如 MA60）之上。</div>
            {/* 模块一：均线参数 */}
            <div className="mt-2 border rounded p-3">
              <div className="text-sm font-medium mb-2">均线参数</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div>
                  <div>短期平均线的天数（默认 5）</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond2MA.above_maN_window} onChange={e => setBCond2MA(p => ({...p, above_maN_window: parseInt(e.target.value||'5',10)}))} disabled={computingB || !bCond2MA.enabled} />
                </div>
                <div>
                  <div>长期平均线的天数（默认 60）</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond2MA.long_ma_days} onChange={e => setBCond2MA(p => ({...p, long_ma_days: parseInt(e.target.value||'60',10)}))} disabled={computingB || !bCond2MA.enabled} />
                </div>
              </div>
            </div>
            {/* 模块二：上方判定规则 */}
            <div className="mt-2 border rounded p-3">
              <div className="text-sm font-medium mb-2">上方判定规则</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div className="flex items-center gap-2">
                  <label className="text-sm">是否必须“连续在上方”</label>
                  <input type="checkbox" className="ml-1" checked={bCond2MA.above_maN_consecutive} onChange={e => setBCond2MA(p => ({...p, above_maN_consecutive: e.target.checked}))} disabled={computingB || !bCond2MA.enabled} />
                </div>
                {bCond2MA.above_maN_consecutive ? (
                  <>
                    <div>
                      <div>需要在长期线上方的天数</div>
                      <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond2MA.above_maN_days} onChange={e => setBCond2MA(p => ({...p, above_maN_days: parseInt(e.target.value||'15',10)}))} disabled={computingB || !bCond2MA.enabled} />
                    </div>
                  </>
                ) : (
                  <>
                    <div>
                      <div>在上方的比例（%）</div>
                      <input type="number" className="mt-1 px-2 py-1 border rounded w-full" min={0} max={100} step={1}
                             value={bCond2MA.above_maN_ratio ?? 80}
                             onChange={e => setBCond2MA(p => ({...p, above_maN_ratio: parseFloat(e.target.value||'80')}))}
                             disabled={computingB || !bCond2MA.enabled} />
                    </div>
                  </>
                )}
              </div>
              <div className="mt-1 text-[11px] text-gray-500 leading-4">提示：
                - 开启“连续在上方”时，使用“需要在上方的天数”。
                - 关闭时，按“A→B 期间短均线在长均线上方的比例”判定，例如 90% 表示至少 90% 的交易日 MA{bCond2MA.above_maN_window} 在 MA{bCond2MA.long_ma_days} 之上。
              </div>
            </div>
          </div>

          {/* 条件3：接近长期线 + 阴线/收≤昨收（无量能上限） */}
          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件3：接近长期线 + 阴线/收不高于昨收</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond2.enabled} onChange={e => setBCond2(p => ({...p, enabled: e.target.checked}))} disabled={computingB} /><span>启用</span></label>
            </div>
            {/* 模块一：触及长期线的比较设置 */}
            <div className="mt-2 border rounded p-3">
              <div className="text-sm font-medium mb-2">比较与长期线</div>
              <div className="text-xs text-gray-500 mb-2">说明：选择用“最低价”或“收盘价”与长期平均线作比较，并设定比较关系（触碰或必须低于）。同时可自定义长期平均线的天数（默认 60）。</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div>
                  <div>用于比较的价格（最低价/收盘价）</div>
                  <select className="mt-1 px-2 py-1 border rounded w-full" value={bCond2.touch_price} onChange={e => setBCond2(p => ({...p, touch_price: e.target.value as any}))} disabled={computingB}>
                    <option value="low">最低</option>
                    <option value="close">收盘</option>
                  </select>
                </div>
                <div>
                  <div>与长期平均线的关系</div>
                  <select className="mt-1 px-2 py-1 border rounded w-full" value={bCond2.touch_relation} onChange={e => setBCond2(p => ({...p, touch_relation: e.target.value as any}))} disabled={computingB}>
                    <option value="le">触碰或等于</option>
                    <option value="lt">必须低于</option>
                  </select>
                </div>
                <div>
                  <div>长期平均线的天数（默认 60）</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond2.long_ma_days}
                         onChange={e => setBCond2(p => ({...p, long_ma_days: parseInt(e.target.value||'60',10)}))}
                         disabled={computingB} />
                </div>
              </div>
            </div>

            {/* 模块二：K线形态要求 */}
            <div className="mt-2 border rounded p-3">
              <div className="text-sm font-medium mb-2">K线形态</div>
              <div className="text-xs text-gray-500 mb-2">说明：可要求“今天为下跌K线（收盘价低于开盘价）”，以及“今天收盘价不高于昨天”，帮助过滤掉偏强的反弹日。</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div className="flex items-center gap-2 col-span-2">
                  <input type="checkbox" className="mr-2" checked={bCond2.require_bearish} onChange={e => setBCond2(p => ({...p, require_bearish: e.target.checked}))} disabled={computingB} />
                  <span>今天是下跌K线（收盘价低于开盘价）</span>
                </div>
                <div className="flex items-center gap-2 col-span-2">
                  <input type="checkbox" className="mr-2" checked={bCond2.require_close_le_prev} onChange={e => setBCond2(p => ({...p, require_close_le_prev: e.target.checked}))} disabled={computingB} />
                  <span>今天收盘价不高于昨天</span>
                </div>
              </div>
            </div>

          </div>

          {/* 条件4：量能上限（相对近期最大量） */}
          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件4：量能上限</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond4VR.enabled} onChange={e => setBCond4VR(p => ({...p, enabled: e.target.checked}))} disabled={computingB} /><span>启用</span></label>
            </div>
            <div className="mt-1 text-xs text-gray-500">
              说明：限制“今天的成交量”相对“近N日最大量”的倍数（VR1）不过高，避免把放量大波动日当作B点。
              VR1 = 今天成交量 / 近N日最大量（不含今天）。
            </div>
            <div className="mt-1 text-[11px] text-gray-500 leading-5">
              计算示例：若参照天数 N=10，近10日最大量=1.00亿股，今天=1.20亿股，则 VR1=1.20。
              当设定阈值 vr1_max=1.2 时通过；若设为 1.1 则不通过。注意参照的“近10日最大量”不包含今天。
            </div>
            {/* 模块一：相对上限倍数 */}
            <div className="mt-2 border rounded p-3">
              <div className="text-sm font-medium mb-2">相对上限倍数</div>
              <div className="text-xs text-gray-500 mb-2">说明：把“最近一段时间里的最大成交量”作为参照，限制今天的成交量不超过该参照的若干倍。留空则不限制。</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div>
                  <div>成交量相对上限（与最近高量对比）</div>
                  <input type="number" step="0.01" className="mt-1 px-2 py-1 border rounded w-full" value={bCond4VR.vr1_max} onChange={e => setBCond4VR(p => ({...p, vr1_max: e.target.value === '' ? '' : parseFloat(e.target.value)}))} disabled={computingB} placeholder="留空不限制"/>
                </div>
              </div>
            </div>
            {/* 模块二：参照期设定 */}
            <div className="mt-2 border rounded p-3">
              <div className="text-sm font-medium mb-2">参照期设定</div>
              <div className="text-xs text-gray-500 mb-2">说明：设定“最近一段时间”的长度，用于计算参照的最大成交量（默认 10 天）。</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div>
                  <div>参照的天数（默认 10）</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond4VR.recent_max_vol_window}
                         onChange={e => setBCond4VR(p => ({...p, recent_max_vol_window: parseInt(e.target.value||'10',10)}))}
                         disabled={computingB} />
                </div>
              </div>
            </div>
          </div>

          {/* 条件5：缩量 */}
          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件5：缩量</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond3.enabled} onChange={e => setBCond3(p => ({...p, enabled: e.target.checked}))} disabled={computingB} /><span>启用</span></label>
            </div>
            {/* 子模块1：非放量（VR1 ≤ 阈值） */}
            <div className="mt-2 border rounded p-3">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium">子模块1：非放量（VR1）</div>
                <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={(bCond3 as any).vr1_enabled ?? false} onChange={e => setBCond3(p => ({...p, vr1_enabled: e.target.checked}))} disabled={computingB} /><span>启用</span></label>
              </div>
              <div className="mt-1 text-xs text-gray-500">说明：与 C 点“放量VR1”相反，这里要求 VR1 ≤ 阈值（今天量不超过近N日最大量的若干倍）。</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div>
                  <div>参照天数 N（默认 10）</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={(bCond3 as any).recent_n ?? 10} onChange={e => setBCond3(p => ({...p, recent_n: parseInt(e.target.value||'10',10)}))} disabled={computingB || !(bCond3 as any).vr1_enabled} />
                </div>
                <div>
                  <div>VR1 阈值倍数（默认 1.2）</div>
                  <input type="number" step="0.01" className="mt-1 px-2 py-1 border rounded w-full" value={(bCond3 as any).vr1_max ?? 1.2} onChange={e => setBCond3(p => ({...p, vr1_max: parseFloat(e.target.value||'1.2')}))} disabled={computingB || !(bCond3 as any).vr1_enabled} />
                </div>
              </div>
            </div>
            {/* 子模块2：量均比较（短≤长） */}
            <div className="mt-2 border rounded p-3">
              <div className="flex items-center justify-between"><div className="text-sm font-medium">子模块2：量均比较（短≤长）</div><label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={(bCond3 as any).vma_rel_enabled ?? false} onChange={e => setBCond3(p => ({...p, vma_rel_enabled: e.target.checked}))} disabled={computingB || !bCond3.enabled} /><span>启用</span></label></div>
              <div className="mt-1 text-xs text-gray-500">说明：与 C 点“短量均&gt;长量均”相反，这里要求短期量均≤长期量均。</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                
                <div>
                  <div>短天数（默认 5）</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond3.short_days} onChange={e => setBCond3(p => ({...p, short_days: parseInt(e.target.value||'5',10)}))} disabled={computingB || !bCond3.enabled || !(bCond3 as any).vma_rel_enabled} />
                </div>
                <div>
                  <div>长天数（默认 10）</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond3.long_days} onChange={e => setBCond3(p => ({...p, long_days: parseInt(e.target.value||'10',10)}))} disabled={computingB || !bCond3.enabled || !(bCond3 as any).vma_rel_enabled} />
                </div>
              </div>
            </div>
            {/* 子模块3：近X日量连降（严格递减） */}
            <div className="mt-2 border rounded p-3">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium">子模块3：近X日量连降</div>
                <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={(bCond3 as any).vol_down_enabled ?? false} onChange={e => setBCond3(p => ({...p, vol_down_enabled: e.target.checked}))} disabled={computingB || !bCond3.enabled} /><span>启用</span></label>
              </div>
              <div className="mt-1 text-xs text-gray-500">说明：要求最近X日（含当日）成交量严格递减，体现缩量走稳。</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div>
                  <div>X（日）</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={(bCond3 as any).vol_decreasing_days ?? 3} onChange={e => setBCond3(p => ({...p, vol_decreasing_days: parseInt(e.target.value||'3',10)}))} disabled={computingB || !bCond3.enabled || !(bCond3 as any).vol_down_enabled} />
                </div>
              </div>
            </div>
          </div>

          {/* 条件6：价稳 */}
          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件6：价稳</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond4.enabled} onChange={e => setBCond4(p => ({...p, enabled: e.target.checked}))} disabled={computingB} /><span>启用</span></label>
            </div>
            <div className="mt-1 text-xs text-gray-500">
              说明：希望“今天的最低价”相对这段时间的“前期最低价”比较稳定。
              你可以选三种方式：
              1）未创新低（今天的最低价不低于这段时间里的最低水平）；
              2）允许有一定幅度的回落（例如不超过 X%）；
              3）按“过去 N 天的典型波动”给一点缓冲，再判断是否稳定（适合有波动的股票）。
            </div>
            <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div>
                <div>判断方式</div>
                <select className="mt-1 px-2 py-1 border rounded w-full" value={bCond4.price_stable_mode} onChange={e => setBCond4(p => ({...p, price_stable_mode: e.target.value as any}))} disabled={computingB}>
                  <option value="no_new_low">未创新低</option>
                  <option value="ratio">允许一定幅度的回落</option>
                  <option value="atr">按“过去的波动”给缓冲</option>
                </select>
              </div>
              <div>
                <div>最多允许回落的比例（例：0.03 表示 3%）</div>
                <input type="number" step="0.01" className="mt-1 px-2 py-1 border rounded w-full" value={bCond4.max_drop_ratio} onChange={e => setBCond4(p => ({...p, max_drop_ratio: parseFloat(e.target.value||'0')}))} disabled={computingB || bCond4.price_stable_mode!=='ratio'} />
              </div>
              <div>
                <div>参考过去多少天的波动</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond4.use_atr_window} onChange={e => setBCond4(p => ({...p, use_atr_window: parseInt(e.target.value||'14',10)}))} disabled={computingB || bCond4.price_stable_mode!=='atr'} />
              </div>
              <div>
                <div>给多少倍的“波动缓冲”</div>
                <input type="number" step="0.01" className="mt-1 px-2 py-1 border rounded w-full" value={bCond4.atr_buffer} onChange={e => setBCond4(p => ({...p, atr_buffer: parseFloat(e.target.value||'0.5')}))} disabled={computingB || bCond4.price_stable_mode!=='atr'} />
              </div>
            </div>
          </div>

          <div>
            <button
              className="px-3 py-2 text-sm font-medium text-white bg-emerald-600 hover:bg-emerald-700 rounded disabled:opacity-50"
              onClick={async () => {
                if (!symbol) return;
                try {
                  setComputingB(true);
                  const body: any = {
                    cond1: { enabled: bCond1.enabled, min_days_from_a: bCond1.min_days_from_a, max_days_from_a: bCond1.max_days_from_a === '' ? null : bCond1.max_days_from_a, allow_multi_b_per_a: bCond1.allow_multi_b_per_a },
                    cond2: {
                      enabled: bCond2MA.enabled,
                      above_maN_window: bCond2MA.above_maN_window,
                      above_maN_days: bCond2MA.above_maN_days,
                      above_maN_consecutive: bCond2MA.above_maN_consecutive,
                      max_maN_below_days: bCond2MA.max_maN_below_days,
                      long_ma_window: bCond2MA.long_ma_days,
                      above_maN_ratio: (!bCond2MA.above_maN_consecutive && bCond2MA.above_maN_ratio != null) ? (bCond2MA.above_maN_ratio / 100) : undefined,
                    },
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
        </div>
      </div>

      {computingB && (
        <div className="fixed inset-0 bg-black/30 z-50 flex items-center justify-center">
          <div className="bg-white rounded px-6 py-4 shadow text-sm">正在计算B点，请稍候...</div>
        </div>
      )}

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

      {/* 寻找C点（基于 B→C 条件组合） */}
      <div className="mt-6 bg-white p-4 rounded shadow">
        <div className="font-semibold mb-2">寻找C点（基于 B→C 条件组合）</div>
        <div className="grid grid-cols-1 gap-3">
          {/* 条件1：时间窗口 */}
          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件1：时间窗口</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond1.enabled} onChange={e => setCCond1(p => ({...p, enabled: e.target.checked}))} disabled={computingC} /><span>启用</span></label>
            </div>
            <div className="text-xs text-gray-500 mt-1">说明：C点距离“最近的B点”不超过指定天数（默认120天）。</div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mt-2">
              <div>
                <div>最大允许天数（默认 60）</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond1.max_days_from_b} onChange={e => setCCond1(p => ({...p, max_days_from_b: parseInt(e.target.value||'60',10)}))} disabled={computingC} />
              </div>
            </div>
          </div>

          {/* 条件2：放量（多模块独立开关） */}
          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件2：放量</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond2.enabled} onChange={e => setCCond2(p => ({...p, enabled: e.target.checked}))} disabled={computingC} /><span>启用</span></label>
            </div>
            {/* 模块一：VR1 放量（默认启用） */}
            <div className="mt-2 border rounded p-3">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium">模块一：当日成交量 ≥ 倍数 × 前N日最大量</div>
                <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond2.vr1_enabled} onChange={e => setCCond2(p => ({...p, vr1_enabled: e.target.checked}))} disabled={computingC || !cCond2.enabled} /><span>启用</span></label>
              </div>
              <div className="text-xs text-gray-500 mt-1">默认 N=10，倍数=2。关闭则不以此作为必要条件。</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mt-2">
                <div>
                  <div>前 N 日（默认 10）</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond2.recent_n} onChange={e => setCCond2(p => ({...p, recent_n: parseInt(e.target.value||'10',10)}))} disabled={computingC || !cCond2.enabled || !cCond2.vr1_enabled} />
                </div>
                <div>
                  <div>倍数阈值（默认 2）</div>
                  <input type="number" step="0.01" className="mt-1 px-2 py-1 border rounded w-full" value={cCond2.vol_multiple} onChange={e => setCCond2(p => ({...p, vol_multiple: parseFloat(e.target.value||'2')}))} disabled={computingC || !cCond2.enabled || !cCond2.vr1_enabled} />
                </div>
              </div>
            </div>

            {/* 模块二：短/长量均线比较 */}
            <div className="mt-2 border rounded p-3">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium">模块二：短期量均线在长期量均线上方</div>
                <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond2.vma_cmp_enabled} onChange={e => setCCond2(p => ({...p, vma_cmp_enabled: e.target.checked}))} disabled={computingC || !cCond2.enabled} /><span>启用</span></label>
              </div>
              <div className="text-xs text-gray-500 mt-1">默认 短=5 日，长=10 日。关闭则不以此作为必要条件。</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mt-2">
                <div>
                  <div>短期量均线（N，默认 5）</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond2.vma_short_days} onChange={e => setCCond2(p => ({...p, vma_short_days: parseInt(e.target.value||'5',10)}))} disabled={computingC || !cCond2.enabled || !cCond2.vma_cmp_enabled} />
                </div>
                <div>
                  <div>长期量均线（M，默认 10）</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond2.vma_long_days} onChange={e => setCCond2(p => ({...p, vma_long_days: parseInt(e.target.value||'10',10)}))} disabled={computingC || !cCond2.enabled || !cCond2.vma_cmp_enabled} />
                </div>
              </div>
            </div>

            {/* 模块三：成交量连升 */}
            <div className="mt-2 border rounded p-3">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium">模块三：前 X 日（含当日）成交量均上升</div>
                <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond2.vol_up_enabled} onChange={e => setCCond2(p => ({...p, vol_up_enabled: e.target.checked}))} disabled={computingC || !cCond2.enabled} /><span>启用</span></label>
              </div>
              <div className="text-xs text-gray-500 mt-1">默认 X=3。要求近 X 日每天的成交量较前一日均更高（严格上升）。</div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mt-2">
                <div>
                  <div>X（日数，默认 3）</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond2.vol_increasing_days} onChange={e => setCCond2(p => ({...p, vol_increasing_days: parseInt(e.target.value||'3',10)}))} disabled={computingC || !cCond2.enabled || !cCond2.vol_up_enabled} />
                </div>
              </div>
            </div>
          </div>

          {/* 条件3：价格与均线 */}
          <div className="border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="font-medium">条件3：价格与均线</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond3.enabled} onChange={e => setCCond3(p => ({...p, enabled: e.target.checked}))} disabled={computingC} /><span>启用</span></label>
            </div>
            <div className="text-xs text-gray-500 mt-1">说明：选择用“收盘/最高/最低”与 MA(Y) 比较，要求在均线上方。</div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mt-2">
              <div>
                <div>用于比较的价格</div>
                <select className="mt-1 px-2 py-1 border rounded w-full" value={cCond3.price_field} onChange={e => setCCond3(p => ({...p, price_field: e.target.value as any}))} disabled={computingC}>
                  <option value="close">收盘价</option>
                  <option value="high">最高价</option>
                  <option value="low">最低价</option>
                </select>
              </div>
              <div>
                <div>均线天数（默认 60）</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond3.ma_days} onChange={e => setCCond3(p => ({...p, ma_days: parseInt(e.target.value||'60',10)}))} disabled={computingC} />
              </div>
              <div>
                <div>比较关系</div>
                <select className="mt-1 px-2 py-1 border rounded w-full" value={cCond3.relation} onChange={e => setCCond3(p => ({...p, relation: e.target.value as any}))} disabled={computingC}>
                  <option value="ge">≥</option>
                  <option value="gt">＞</option>
                </select>
              </div>
            </div>
          </div>

          {/* 操作 */}
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
      </div>
    </div>

    {computingC && (
      <div className="fixed inset-0 bg-black/30 z-50 flex items-center justify-center" aria-live="polite" aria-busy>
        <div className="bg-white rounded px-6 py-4 shadow text-sm">正在计算C点，请稍候...</div>
      </div>
    )}

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
