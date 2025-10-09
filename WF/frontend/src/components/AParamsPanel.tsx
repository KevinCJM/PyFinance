import React from 'react';
import type { ACond1, ACond2, ACond3, ACondVol } from '../hooks/useABCDefaults';

export default function AParamsPanel({
  aCond1, setACond1,
  aCond2, setACond2,
  aCond3, setACond3,
  aCondVol, setACondVol,
  computingA,
  actions,
}: {
  aCond1: ACond1; setACond1: React.Dispatch<React.SetStateAction<ACond1>>;
  aCond2: ACond2; setACond2: React.Dispatch<React.SetStateAction<ACond2>>;
  aCond3: ACond3; setACond3: React.Dispatch<React.SetStateAction<ACond3>>;
  aCondVol: ACondVol; setACondVol: React.Dispatch<React.SetStateAction<ACondVol>>;
  computingA: boolean;
  actions?: React.ReactNode;
}) {
  return (
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
                - 开启：你设定的每条短期线，都需要在“上穿完备窗口”的天数里至少出现过一次“从下
                向上的越过”；且在这个信号当天，这些短期线都处于长期线上方。
                - 关闭：只要其中任意一条短期线在当天出现“从下向上越过长期线”，就可以触发。
              </div>
            </div>
          </div>
        </div>

        {/* 条件3：价格上穿确认 */}
        <div className="border rounded p-3">
          <div className="flex items-center justify-between">
            <div className="font-medium">条件3：价格上穿确认</div>
            <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={aCond3.启用} onChange={e => setACond3(p => ({...p, 启用: e.target.checked}))} disabled={computingA} /><span>启用</span></label>
          </div>
          <div className="mt-1 text-xs text-gray-500">
            说明：为了进一步确认，我们设定在 A 点出现之后的“前置一定天数”里，价格（可以选择“最高价”或“收盘价”）至少出现一次“从下方向上越过设定的“确认均线”，这有助于减少偶然波动带来的误判。
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
              <select className="mt-1 px-2 py-1 border rounded w-full" value={aCond3.确认价格列} onChange={e => setACond3(p => ({...p, 确认价格列: e.target.value as 'open'|'high'|'low'|'close'}))} disabled={computingA}>
                <option value="close">收盘价</option>
                <option value="high">最高价</option>
                <option value="low">最低价</option>
              </select>
            </div>
          </div>
        </div>

        {/* 条件4：放量确认（模块化） */}
        <div className="border rounded p-3">
          <div className="font-medium mb-2">条件4：放量确认（模块化）</div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
            {/* 子模块1：VR1 放量 */}
            <div className="border rounded p-3">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium">子模块1：VR1 放量</div>
                <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={aCondVol.vr1_enabled} onChange={e => setACondVol(p => ({...p, vr1_enabled: e.target.checked}))} disabled={computingA || !aCondVol.启用} /><span>启用</span></label>
              </div>
              <div className="mt-1 text-xs text-gray-500">说明：与 C 点“放量VR1”相同，这里要求 VR1 ≥ 阈值（今天量 ≥ 近 N 日最大量的若干倍）。</div>
              <div className="mt-2 grid grid-cols-2 gap-3">
                <div>
                  <div>回看天数</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={aCondVol.对比天数} onChange={e => setACondVol(p => ({...p, 对比天数: parseInt(e.target.value||'10',10)}))} disabled={computingA || !aCondVol.启用 || !aCondVol.vr1_enabled} />
                </div>
                <div>
                  <div>倍数阈值</div>
                  <input type="number" step="0.01" className="mt-1 px-2 py-1 border rounded w-full" value={aCondVol.倍数} onChange={e => setACondVol(p => ({...p, 倍数: parseFloat(e.target.value||'2')}))} disabled={computingA || !aCondVol.启用 || !aCondVol.vr1_enabled} />
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

        {actions}
      </div>
    </div>
  );
}

