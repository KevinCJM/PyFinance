import React from 'react'
import ReactECharts from 'echarts-for-react'

type KLineData = {
  日期: string
  开盘: number
  收盘: number
  最高: number
  最低: number
  成交量: number
}

type Point = { date: string; price_low?: number|null; price_close?: number|null }

type AuxLine = {
  id?: string
  name: string
  type: '普通连线'|'移动均线'
  field: '开盘'|'收盘'|'最高'|'最低'
  window?: number
}

type VolAuxLine = { id?: string; name: string; window: number }

export default function KLineEChart({
  symbol,
  klineData,
  aPoints = [],
  bPoints = [],
  cPoints = [],
  auxLines = [],
  volAuxLines = [],
  style,
}: {
  symbol: string
  klineData: KLineData[]
  aPoints?: Point[]
  bPoints?: Point[]
  cPoints?: Point[]
  auxLines?: AuxLine[]
  volAuxLines?: VolAuxLine[]
  style?: React.CSSProperties
}) {
  const getOption = React.useCallback(() => {
    if (!klineData || !klineData.length) return {}
    const dates = klineData.map(d => d['日期'])
    const toNum = (v: any): number|null => (v == null || v === '' || isNaN(Number(v)) ? null : Number(v))

    const data = klineData.map(item => {
      let o = toNum(item['开盘'])
      let c = toNum(item['收盘'])
      let lo = toNum(item['最低'])
      let hi = toNum(item['最高'])
      if (lo == null && (o != null || c != null)) lo = Math.min(o ?? Infinity, c ?? Infinity)
      if (hi == null && (o != null || c != null)) hi = Math.max(o ?? -Infinity, c ?? -Infinity)
      if (lo != null && hi != null && lo > hi) { const t = lo; lo = hi; hi = t }
      return [o, c, lo, hi]
    })

    const volumes = klineData.map((item, index) => [
      index,
      item['成交量'],
      item['开盘'] > item['收盘'] ? -1 : 1,
    ])

    const legendData: string[] = []
    const extraSeries: any[] = []

    const numericFromField = (field: '开盘'|'收盘'|'最高'|'最低') => klineData.map(d => toNum((d as any)[field]))
    const ma = (arr: number[], win: number) => {
      const out: (number|null)[] = new Array(arr.length).fill(null)
      if (win <= 1) return arr.slice()
      let sum = 0
      for (let i=0; i<arr.length; i++) {
        const v = arr[i]; sum += v
        if (i >= win) sum -= arr[i - win]
        if (i >= win - 1) out[i] = +(sum / win)
      }
      return out
    }

    for (const line of auxLines) {
      legendData.push(line.name)
      let y: (number|null)[] | number[] = []
      if (line.type === '普通连线') {
        y = numericFromField(line.field) as (number|null)[]
      } else {
        const src = numericFromField(line.field) as number[]
        const w = Math.max(1, line.window || 5)
        y = ma(src, w)
      }
      extraSeries.push({ name: line.name, type: 'line', data: y, yAxisIndex: 0, xAxisIndex: 0, showSymbol: false, connectNulls: true, smooth: false, lineStyle: { width: 1.5 }, emphasis: { focus: 'series' } })
    }

    const volArr: number[] = klineData.map(item => toNum(item['成交量']) || 0)
    const maVol = (arr: number[], win: number) => {
      const out: (number|null)[] = new Array(arr.length).fill(null)
      if (win <= 1) return arr.slice()
      let sum = 0
      for (let i=0; i<arr.length; i++) { sum += arr[i]; if (i >= win) sum -= arr[i-win]; if (i >= win-1) out[i] = +(sum / win) }
      return out
    }
    for (const vline of volAuxLines) {
      legendData.push(vline.name)
      const w = Math.max(1, vline.window || 5)
      const y = maVol(volArr, w)
      extraSeries.push({ name: vline.name, type: 'line', data: y, xAxisIndex: 1, yAxisIndex: 1, showSymbol: false, connectNulls: true, smooth: false, lineStyle: { width: 1.2 }, emphasis: { focus: 'series' } })
    }

    const dateToIndex = new Map<string, number>()
    dates.forEach((d, i) => dateToIndex.set(d, i))

    const pushScatter = (name: string, pts: Point[], yMul: number, color: string) => {
      if (!pts || !pts.length) return
      legendData.push(name)
      const xs: string[] = []
      const ys: number[] = []
      for (const p of pts) {
        const di = dateToIndex.get(p.date)
        if (di !== undefined) {
          const base = typeof p.price_low === 'number' ? p.price_low! : (typeof p.price_close === 'number' ? p.price_close! : null)
          if (base != null && !Number.isNaN(base)) { xs.push(p.date); ys.push(+(+base * yMul).toFixed(2)) }
        }
      }
      extraSeries.push({ name, type: 'scatter', data: xs.map((d,i)=>[d, ys[i]]), xAxisIndex: 0, yAxisIndex: 0, symbol: name==='A点'?'triangle':(name==='B点'?'diamond':'circle'), symbolSize: 12, itemStyle: { color }, tooltip: { valueFormatter: (v:any)=> (v==null?'-':(+v).toFixed(2)) } })
    }

    pushScatter('A点', aPoints, 0.995, '#2563eb')
    pushScatter('B点', bPoints, 1.005, '#10b981')
    pushScatter('C点', cPoints, 1.01, '#f59e0b')

    return {
      title: { text: `${symbol} K线图`, left: 'center' },
      tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
      legend: { show: (auxLines.length + volAuxLines.length + aPoints.length + bPoints.length + cPoints.length) > 0, data: legendData, top: 30 },
      grid: [ { left: '10%', right: '8%', height: '50%' }, { left: '10%', right: '8%', top: '65%', height: '16%' } ],
      xAxis: [ { type: 'category', data: dates, scale: true, boundaryGap: false, axisLine: { onZero: false }, splitLine: { show: false }, min: 'dataMin', max: 'dataMax' }, { type: 'category', gridIndex: 1, data: dates, scale: true, boundaryGap: false, axisLine: { onZero: false }, axisTick: { show: false }, splitLine: { show: false }, axisLabel: { show: false }, min: 'dataMin', max: 'dataMax' } ],
      yAxis: [ { scale: true, splitArea: { show: true }, axisLabel: { formatter: (v:number)=> (v==null?'':(+v).toFixed(2)) } }, { scale: true, gridIndex: 1, splitNumber: 2, axisLabel: { show: true, formatter: (v:number)=> (v==null?'':(+v).toFixed(2)) }, axisLine: { show: false }, axisTick: { show: false }, splitLine: { show: false } } ],
      dataZoom: [ { type: 'inside', xAxisIndex: [0,1], start: 80, end: 100 }, { show: true, xAxisIndex: [0,1], type: 'slider', top: '85%', start: 80, end: 100 } ],
      series: [
        { name: 'K线', type: 'candlestick', data, itemStyle: { color: '#ec0000', color0: '#00da3c', borderColor: '#8A0000', borderColor0: '#008F28' } },
        { name: '成交量', type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: volumes.map(v=>v[1]), itemStyle: { color: ({ dataIndex }: any) => { const o = Number(klineData[dataIndex]['开盘']); const c = Number(klineData[dataIndex]['收盘']); return o > c ? '#00da3c' : '#ec0000' } }, tooltip: { valueFormatter: (v:any)=>(v==null?'-':(+v).toFixed(2)) } },
        ...extraSeries,
      ]
    }
  }, [symbol, klineData, aPoints, bPoints, cPoints, auxLines, volAuxLines])

  return <ReactECharts option={getOption()} style={style || { height: '600px' }} />
}
