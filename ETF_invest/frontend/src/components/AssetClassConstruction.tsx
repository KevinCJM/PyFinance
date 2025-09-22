import React, { useMemo, useState, useEffect } from 'react'

type WeightMode = 'custom' | 'equal' | 'risk'
type RiskMetric = 'vol' | 'var' | 'es'

interface ETFItem {
  code: string
  name: string
  weight?: number
  riskContribution?: number
  solved?: boolean
}

interface AssetClass {
  id: string
  name: string
  mode: WeightMode
  etfs: ETFItem[]
  riskMetric?: RiskMetric
  maxLeverage?: number
}

const ETF_UNIVERSE: { code: string; name: string }[] = [
  { code: '510300', name: '沪深300ETF' },
  { code: '510500', name: '中证500ETF' },
  { code: '159919', name: '沪深300ETF 易方达' },
  { code: '159922', name: '中证500ETF 嘉实' },
  { code: '510050', name: '上证50ETF' },
  { code: '159915', name: '创业板ETF' },
  { code: 'TLT', name: 'iShares 20+ Year Treasury' },
  { code: 'IEF', name: 'iShares 7-10 Year Treasury' },
  { code: 'SHY', name: 'iShares 1-3 Year Treasury' },
  { code: 'AGG', name: 'iShares Core US Aggregate Bond' },
  { code: 'LQD', name: 'iShares iBoxx $ Inv Grade Corp Bd' },
  { code: 'HYG', name: 'iShares iBoxx $ High Yield Corp Bd' },
  { code: 'SPY', name: 'SPDR S&P 500 ETF Trust' },
  { code: 'QQQ', name: 'Invesco QQQ Trust' },
]

function fuzzySearchTop(q: string, k = 10) {
  const query = q.trim().toLowerCase()
  if (!query) return ETF_UNIVERSE.slice(0, k)
  const scored = ETF_UNIVERSE
    .map((x) => {
      const hay = (x.code + ' ' + x.name).toLowerCase()
      const idx = hay.indexOf(query)
      const score = idx === -1 ? Infinity : idx + query.length * 0.2
      return { x, score }
    })
    .filter((s) => s.score !== Infinity)
    .sort((a, b) => a.score - b.score)
    .slice(0, k)
    .map((s) => s.x)
  return scored
}

function uid() {
  return Math.random().toString(36).slice(2, 10)
}

function clamp(n: number, a: number, b: number) {
  return Math.max(a, Math.min(b, n))
}

function round2(n: number) {
  return Math.round(n * 100) / 100
}

function equalWeights(n: number): number[] {
  if (n <= 0) return []
  const w = 100 / n
  const arr = Array(n).fill(Math.floor(w * 100) / 100) as number[]
  let rem = round2(100 - arr.reduce((a, b) => a + b, 0))
  let i = 0
  while (rem > 0 && i < n) {
    arr[i] = round2(arr[i] + 0.01)
    rem = round2(rem - 0.01)
    i++
  }
  return arr
}

export default function AssetClassConstructionPage() {
  const [classes, setClasses] = useState<AssetClass[]>([
    {
      id: uid(),
      name: '权益类',
      mode: 'custom',
      etfs: [
        { code: 'SPY', name: 'SPDR S&P 500 ETF Trust', weight: 80 },
        { code: 'QQQ', name: 'Invesco QQQ Trust', weight: 20 },
      ],
      riskMetric: 'vol',
      maxLeverage: 0,
    },
    {
      id: uid(),
      name: '固收类',
      mode: 'equal',
      etfs: [
        { code: 'TLT', name: 'iShares 20+ Year Treasury' },
        { code: 'IEF', name: 'iShares 7-10 Year Treasury' },
      ],
      riskMetric: 'vol',
      maxLeverage: 0,
    },
  ])

  const [loading, setLoading] = useState(false)
  const [searchOpen, setSearchOpen] = useState<{ open: boolean; classId?: string }>({ open: false })
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<{ code: string; name: string }[]>([])

  function updateClass(id: string, updater: (c: AssetClass) => AssetClass) {
    setClasses((prev) => prev.map((c) => (c.id === id ? updater(c) : c)))
  }

  function setCustomWeight(classId: string, idx: number, val: number) {
    updateClass(classId, (c) => {
      const etfs = c.etfs.map((e, i) => (i === idx ? { ...e, weight: clamp(val, 0, 100) } : e))
      return { ...c, etfs }
    })
  }

  function setRiskContribution(classId: string, idx: number, val: number) {
    updateClass(classId, (c) => {
      const etfs = c.etfs.map((e, i) => (i === idx ? { ...e, riskContribution: clamp(val, 0, 100) } : e))
      return { ...c, etfs }
    })
  }

  function setMaxLeverage(classId: string, val: number) {
    updateClass(classId, (c) => ({ ...c, maxLeverage: clamp(val, 0, 100) }))
  }

  function addETFToClass(classId: string, etf: { code: string; name: string }) {
    updateClass(classId, (c) => {
      const exists = c.etfs.some((x) => x.code === etf.code && x.name === etf.name)
      const etfs = exists
        ? c.etfs
        : [
            ...c.etfs,
            {
              ...etf,
              weight: c.mode === 'custom' ? 0 : undefined,
              riskContribution: c.mode === 'risk' ? 0 : undefined,
              solved: false,
            },
          ]
      return { ...c, etfs }
    })
  }

  function removeETF(classId: string, idx: number) {
    updateClass(classId, (c) => ({ ...c, etfs: c.etfs.filter((_, i) => i !== idx) }))
  }

  function addAssetClass() {
    setClasses((prev) => [...prev, { id: uid(), name: '新大类', mode: 'custom', etfs: [], riskMetric: 'vol', maxLeverage: 0 }])
  }

  function deleteAssetClass(id: string) {
    setClasses((prev) => prev.filter((c) => c.id !== id))
  }

  async function onSolveRiskWeights(classId: string) {
    const ac = classes.find((c) => c.id === classId)
    if (!ac) return
    if (ac.mode !== 'risk') {
      alert('请先切换到“风险平价”模式')
      return
    }
    const sumRisk = ac.etfs.reduce((a, e) => a + (e.riskContribution ?? 0), 0)
    if (Math.abs(sumRisk - 100) > 1e-6) {
      alert('风险贡献合计需等于 100%，请调整后再计算')
      return
    }
    try {
      setLoading(true)
      const payload = {
        assetClassId: ac.id,
        riskMetric: ac.riskMetric ?? 'vol',
        maxLeverage: ac.maxLeverage ?? 0,
        etfs: ac.etfs.map((e) => ({ code: e.code, name: e.name, riskContribution: e.riskContribution ?? 0 })),
      }
      const resp = await fetch('/api/risk-parity/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!resp.ok) throw new Error(`后端返回错误状态 ${resp.status}`)
      const data: { weights: number[] } = await resp.json()
      if (!Array.isArray(data.weights) || data.weights.length !== ac.etfs.length) {
        throw new Error('返回的权重数量与ETF数量不一致')
      }
      updateClass(classId, (c) => ({
        ...c,
        etfs: c.etfs.map((e, i) => ({ ...e, weight: round2(Math.max(0, data.weights[i])), solved: true })),
      }))
      alert('反推资金权重已计算并写入')
    } catch (err: any) {
      console.error(err)
      alert('计算失败：' + err.message + '\n请确认已启动 Python 后端 (POST /api/risk-parity/solve)。')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    try {
      const ew = equalWeights(3)
      console.assert(ew.length === 3, 'equalWeights 长度应为 3')
      console.assert(Math.abs(ew.reduce((a, b) => a + b, 0) - 100) < 1e-6, 'equalWeights 合计应为 100')
      const riskSum = [{ v: 60 }, { v: 40 }].reduce((a, e) => a + e.v, 0)
      console.assert(riskSum === 100, '手动风险贡献合计应为 100')
      console.assert(clamp(150, 0, 100) === 100, 'clamp 上界应生效')
      console.assert(round2(1.234) === 1.23 && round2(1.235) === 1.24, 'round2 四舍五入应生效')
      console.assert(ETF_UNIVERSE.some((x) => x.code === 'SPY'), '搜索应包含 SPY')
    } catch (e) {
      console.error('内置测试失败:', e)
    }
  }, [])

  // 搜索：优先从后端读取 data 下的 ETF 列表（JSON/Parquet），失败时回退本地模糊匹配
  useEffect(() => {
    const controller = new AbortController()
    const doFetch = async () => {
      try {
        const url = `/api/etf/search?q=${encodeURIComponent(searchQuery)}&k=10`
        const resp = await fetch(url, { signal: controller.signal })
        if (!resp.ok) throw new Error(`status ${resp.status}`)
        const data = await resp.json()
        if (Array.isArray(data)) setSearchResults(data)
        else setSearchResults(fuzzySearchTop(searchQuery, 10))
      } catch {
        setSearchResults(fuzzySearchTop(searchQuery, 10))
      }
    }
    doFetch()
    return () => controller.abort()
  }, [searchQuery])

  return (
    <div className="mx-auto max-w-5xl p-6">
      <h1 className="text-2xl font-semibold">资产大类构建模块</h1>
      <p className="text-sm text-gray-500 mt-1">配置资产大类、ETF 选择与权重；风险平价支持手动风险贡献、最大杠杆与后端反推权重</p>

      <div className="mt-5 rounded-2xl border border-gray-200 bg-white p-4">
        <SectionTitle title="构建资产大类" />
        <div className="space-y-6">
          {classes.map((ac) => (
            <AssetClassCard
              key={ac.id}
              ac={ac}
              on重命名={(name) => updateClass(ac.id, (c) => ({ ...c, name }))}
              onModeChange={(mode) => updateClass(ac.id, (c) => ({ ...c, mode }))}
              onRiskMetricChange={(metric) => updateClass(ac.id, (c) => ({ ...c, riskMetric: metric }))}
              on删除={() => deleteAssetClass(ac.id)}
              onAddETF={() => setSearchOpen({ open: true, classId: ac.id })}
              onRemoveETF={(idx) => removeETF(ac.id, idx)}
              onSetCustomWeight={(idx, v) => setCustomWeight(ac.id, idx, v)}
              onSetRiskContribution={(idx, v) => setRiskContribution(ac.id, idx, v)}
              onSetMaxLeverage={(v) => setMaxLeverage(ac.id, v)}
              onSolve={() => onSolveRiskWeights(ac.id)}
              loading={loading}
            />
          ))}

          <button className="w-full rounded-xl border border-dashed border-gray-300 py-3 text-sm hover:bg-gray-50" onClick={addAssetClass}>
            + 添加新大类
          </button>
        </div>
      </div>

      {searchOpen.open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-xl rounded-2xl bg-white p-5 shadow-xl">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">添加 ETF</h3>
              <button className="text-gray-500" onClick={() => setSearchOpen({ open: false })}>✕</button>
            </div>
            <input
              autoFocus
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="按代码或名称搜索..."
              className="mt-3 w-full rounded-lg border border-gray-300 px-3 py-2 outline-none focus:ring-2 focus:ring-blue-500"
            />
            <div className="mt-3 max-h-80 overflow-auto rounded-lg border border-gray-100">
              {searchResults.length === 0 && <div className="p-4 text-sm text-gray-500">无匹配结果</div>}
              {searchResults.map((etf) => (
                <button
                  key={etf.code + etf.name}
                  onClick={() => {
                    if (searchOpen.classId) addETFToClass(searchOpen.classId, etf)
                    setSearchQuery('')
                    setSearchOpen({ open: false })
                  }}
                  className="flex w-full items-center justify-between border-b px-4 py-2 text-left hover:bg-gray-50"
                >
                  <span className="font-mono text-sm">{etf.code}</span>
                  <span className="ml-3 flex-1 truncate px-4 text-sm text-gray-700">{etf.name}</span>
                  <span className="text-xs text-gray-400">添加</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function SectionTitle({ title }: { title: string }) {
  return (
    <div className="mb-3 flex items-center gap-2">
      <span className="text-blue-600">◆</span>
      <h2 className="text-lg font-semibold">{title}</h2>
    </div>
  )
}

function ModePill({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`rounded-md px-2 py-1 text-xs ${active ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
      style={{ padding: '0.25rem 0.5rem' }}
    >
      {label}
    </button>
  )
}

function AssetClassCard({
  ac,
  on重命名,
  onModeChange,
  onRiskMetricChange,
  on删除,
  onAddETF,
  onRemoveETF,
  onSetCustomWeight,
  onSetRiskContribution,
  onSetMaxLeverage,
  onSolve,
  loading,
}: {
  ac: AssetClass
  on重命名: (name: string) => void
  onModeChange: (mode: WeightMode) => void
  onRiskMetricChange: (metric: RiskMetric) => void
  on删除: () => void
  onAddETF: () => void
  onRemoveETF: (idx: number) => void
  onSetCustomWeight: (idx: number, val: number) => void
  onSetRiskContribution: (idx: number, val: number) => void
  onSetMaxLeverage: (val: number) => void
  onSolve: () => void
  loading: boolean
}) {
  const [editing, setEditing] = useState(false)
  const [tempName, setTempName] = useState(ac.name)
  useEffect(() => setTempName(ac.name), [ac.name])

  const isRisk = ac.mode === 'risk'
  const showSolved = isRisk && ac.etfs.some((e) => e.solved)

  const equal = useMemo(() => (ac.mode === 'equal' ? equalWeights(ac.etfs.length) : []), [ac.mode, ac.etfs.length])

  const sumWeight = useMemo(() => {
    if (ac.mode === 'equal') return 100
    if (ac.mode === 'custom') return round2(ac.etfs.reduce((a, e) => a + (e.weight ?? 0), 0))
    return round2(ac.etfs.reduce((a, e) => a + (e.weight ?? 0), 0))
  }, [ac])

  const sumRisk = useMemo(() => {
    if (!isRisk) return 0
    return round2(ac.etfs.reduce((a, e) => a + (e.riskContribution ?? 0), 0))
  }, [ac, isRisk])

  return (
    <div className="rounded-xl border border-gray-200">
      <div className="flex items-center justify-between border-b bg-gray-50/80 px-3 py-2">
        <div className="flex items-center gap-2">
          {editing ? (
            <input
              value={tempName}
              onChange={(e) => setTempName(e.target.value)}
              onBlur={() => {
                on重命名(tempName.trim() || ac.name)
                setEditing(false)
              }}
              className="rounded-md border border-gray-300 px-2 py-1 text-sm"
            />
          ) : (
            <div className="text-sm font-medium">{ac.name}</div>
          )}
          <button className="text-xs text-blue-600 underline" onClick={() => setEditing((v) => !v)} title="重命名">
            {editing ? '保存' : '重命名'}
          </button>

          <div className="ml-3 flex items-center gap-2">
            <ModePill label="自定义权重" active={ac.mode === 'custom'} onClick={() => onModeChange('custom')} />
            <ModePill label="等权重" active={ac.mode === 'equal'} onClick={() => onModeChange('equal')} />
            <ModePill label="风险平价" active={ac.mode === 'risk'} onClick={() => onModeChange('risk')} />

            {isRisk && (
              <>
                <select
                  className="ml-2 rounded-md border border-gray-300 px-2 py-1 text-xs"
                  value={ac.riskMetric || 'vol'}
                  onChange={(e) => onRiskMetricChange(e.target.value as RiskMetric)}
                >
                  <option value="vol">波动率</option>
                  <option value="var">VaR</option>
                  <option value="es">ES</option>
                </select>

                <div className="ml-2 flex items-center gap-2 text-xs">
                  <span className="text-gray-600">最大杠杆</span>
                  <input
                    type="number"
                    min={0}
                    step={0.01}
                    value={ac.maxLeverage ?? 0}
                    onChange={(e) => onSetMaxLeverage(Number(e.target.value))}
                    className="w-20 rounded-md border border-gray-300 px-2 py-1 text-right"
                    title="允许的组合最大杠杆率，例如 0 表示不允许杠杆；2 表示最多 2x"
                  />
                </div>

                <button
                  className="ml-2 rounded-md bg-blue-600 px-3 py-1 text-xs text-white hover:bg-blue-700 disabled:opacity-60"
                  onClick={onSolve}
                  disabled={loading || sumRisk !== 100}
                  title={sumRisk !== 100 ? '风险贡献合计需等于 100% 才能计算' : ''}
                >
                  {loading ? '计算中...' : '计算反推资金权重'}
                </button>
              </>
            )}
          </div>
        </div>
        <button className="rounded-md border border-red-200 px-3 py-1 text-xs text-red-600 hover:bg-red-50" onClick={on删除}>
          删除这个大类
        </button>
      </div>

      <div className="grid grid-cols-12 items-center gap-2 px-3 py-2 text-xs text-gray-500">
        <div className="col-span-7">ETF</div>
        <div className="col-span-3 text-right">{isRisk ? (showSolved ? '风险贡献（%） / 权重（%）' : '风险贡献（%）') : '权重（%）'}</div>
        <div className="col-span-2 text-right">操作</div>
      </div>

      <div className="divide-y">
        {ac.etfs.map((e, idx) => (
          <div key={idx} className="grid grid-cols-12 items-center gap-2 px-3 py-2">
            <div className="col-span-7">
              <div className="flex items-center gap-2">
                <span className="w-5 text-xs text-gray-500">{idx + 1}.</span>
                <div className="min-w-0 flex-1 truncate text-sm">
                  <span className="font-mono">{e.code}</span>
                  <span className="mx-2 text-gray-400">-</span>
                  <span className="text-gray-700">{e.name}</span>
                </div>
              </div>
            </div>

            <div className="col-span-3 text-right">
              {ac.mode === 'custom' ? (
                <div className="inline-flex items-center gap-2">
                  <input
                    type="number"
                    min={0}
                    max={100}
                    step={0.01}
                    value={e.weight ?? 0}
                    onChange={(ev) => onSetCustomWeight(idx, Number(ev.target.value))}
                    className="w-24 rounded-md border border-gray-300 px-2 py-1 text-right text-sm"
                  />
                  <span className="text-sm text-gray-500">%</span>
                </div>
              ) : ac.mode === 'equal' ? (
                <div className="pr-2 text-sm text-gray-700">{(equal[idx] ?? 0).toFixed(2)}%</div>
              ) : (
                <div className="inline-flex items-center gap-3 justify-end">
                  <input
                    type="number"
                    min={0}
                    max={100}
                    step={0.01}
                    value={e.riskContribution ?? 0}
                    onChange={(ev) => onSetRiskContribution(idx, Number(ev.target.value))}
                    className="w-24 rounded-md border border-gray-300 px-2 py-1 text-right text-sm"
                  />
                  <span className="text-sm text-gray-500">%</span>
                  {showSolved && <span className="text-xs text-gray-500">/ 权重 {(e.weight ?? 0).toFixed(2)}%</span>}
                </div>
              )}
            </div>

            <div className="col-span-2 text-right">
              <button onClick={() => onRemoveETF(idx)} className="rounded-md border border-gray-300 px-3 py-1 text-xs hover:bg-gray-50">
                删除
              </button>
            </div>
          </div>
        ))}

        <div className="px-3 py-2">
          <button onClick={onAddETF} className="w-full rounded-lg border border-dashed border-gray-300 py-2 text-sm hover:bg-gray-50">
            + 添加新的 ETF
          </button>
        </div>
      </div>

      <div className="border-t px-3 py-2 text-xs">
        {ac.mode === 'custom' ? (
          <>
            <span
              className={
                'rounded-md px-2 py-0.5 ' + (sumWeight === 100 ? 'bg-green-50 text-green-700' : 'bg-yellow-50 text-yellow-700')
              }
            >
              权重合计： {sumWeight.toFixed(2)}%
            </span>
            {sumWeight !== 100 && <span className="ml-2 text-yellow-700">（需等于 100%）</span>}
          </>
        ) : ac.mode === 'equal' ? (
          <span className="rounded-md bg-green-50 px-2 py-0.5 text-green-700">权重合计： 100.00%</span>
        ) : (
          <>
            <span
              className={
                'rounded-md px-2 py-0.5 ' + (sumRisk === 100 ? 'bg-green-50 text-green-700' : 'bg-yellow-50 text-yellow-700')
              }
            >
              风险贡献合计： {sumRisk.toFixed(2)}%
            </span>
            {sumRisk !== 100 && <span className="ml-2 text-yellow-700">（需等于 100%）</span>}
          </>
        )}
      </div>
    </div>
  )
}
