//
// eslint-disable-next-line @typescript-eslint/no-unused-vars

export default function CParamsPanel({ cCond1, setCCond1, cCond2, setCCond2, cCond3, setCCond3, computingC, actions }: any) {
  return (
    <div className="mt-6 bg-white p-4 rounded shadow">
      <div className="font-semibold mb-2">寻找C点（基于 B→C 条件组合）</div>
      <div className="grid grid-cols-1 gap-3">
        {/* 条件1：时间窗口 */}
        <div className="border rounded p-3">
          <div className="flex items-center justify-between">
            <div className="font-medium">条件1：时间窗口</div>
            <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond1.enabled} onChange={e => setCCond1((p: any) => ({...p, enabled: e.target.checked}))} disabled={computingC} /><span>启用</span></label>
          </div>
          <div className="text-xs text-gray-500 mt-1">说明：C点距离“最近的B点”不超过指定天数（默认120天）。</div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mt-2">
            <div>
              <div>最大允许天数（默认 60）</div>
              <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond1.max_days_from_b} onChange={e => setCCond1((p: any) => ({...p, max_days_from_b: parseInt(e.target.value||'60',10)}))} disabled={computingC} />
            </div>
          </div>
        </div>

        {/* 条件2：放量确认（模块化） */}
        <div className="border rounded p-3">
          <div className="flex items-center justify-between">
            <div className="font-medium">条件2：放量确认（模块化，与B点条件4一致）</div>
            <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond2.enabled} onChange={e => setCCond2((p: any) => ({...p, enabled: e.target.checked}))} disabled={computingC} /><span>启用</span></label>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm mt-2">
            {/* 子模块1：VR1 放量 */}
            <div className="border rounded p-3">
              <div className="flex items-center justify-between"><div className="text-sm font-medium">VR1 放量</div><label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond2.vr1_enabled} onChange={e => setCCond2((p: any) => ({...p, vr1_enabled: e.target.checked}))} disabled={computingC || !cCond2.enabled} /><span>启用</span></label></div>
              <div className="mt-2 grid grid-cols-2 gap-3">
                <div>
                  <div>回看天数</div>
                  <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond2.recent_n} onChange={e => setCCond2((p: any) => ({...p, recent_n: parseInt(e.target.value||'10',10)}))} disabled={computingC || !cCond2.enabled || !cCond2.vr1_enabled} />
                </div>
                <div>
                  <div>倍数阈值</div>
                  <input type="number" step="0.01" className="mt-1 px-2 py-1 border rounded w-full" value={cCond2.vol_multiple} onChange={e => setCCond2((p: any) => ({...p, vol_multiple: parseFloat(e.target.value||'2')}))} disabled={computingC || !cCond2.enabled || !cCond2.vr1_enabled} />
                </div>
              </div>
            </div>
            {/* 子模块2：量均比较 */}
            <div className="border rounded p-3">
              <div className="flex items-center justify-between"><div className="text-sm font-medium">量均比较</div><label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond2.vma_cmp_enabled} onChange={e => setCCond2((p: any) => ({...p, vma_cmp_enabled: e.target.checked}))} disabled={computingC || !cCond2.enabled} /><span>启用</span></label></div>
              <div className="mt-2 grid grid-cols-2 gap-3"><div><div>短期天数</div><input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond2.vma_short_days} onChange={e => setCCond2((p: any) => ({...p, vma_short_days: parseInt(e.target.value||'5',10)}))} disabled={computingC || !cCond2.enabled || !cCond2.vma_cmp_enabled} /></div><div><div>长期天数</div><input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond2.vma_long_days} onChange={e => setCCond2((p: any) => ({...p, vma_long_days: parseInt(e.target.value||'10',10)}))} disabled={computingC || !cCond2.enabled || !cCond2.vma_cmp_enabled} /></div></div>
            </div>
            {/* 子模块3：近X日量连升 */}
            <div className="border rounded p-3">
              <div className="flex items-center justify-between"><div className="text-sm font-medium">近X日量连升</div><label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond2.vol_up_enabled} onChange={e => setCCond2((p: any) => ({...p, vol_up_enabled: e.target.checked}))} disabled={computingC || !cCond2.enabled} /><span>启用</span></label></div>
              <div className="mt-2 grid grid-cols-1 gap-3"><div><div>X（日）</div><input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond2.vol_increasing_days} onChange={e => setCCond2((p: any) => ({...p, vol_increasing_days: parseInt(e.target.value||'3',10)}))} disabled={computingC || !cCond2.enabled || !cCond2.vol_up_enabled} /></div></div>
            </div>
          </div>
        </div>

        {/* 条件3：价格与均线 */}
        <div className="border rounded p-3">
          <div className="flex items-center justify之间"><div className="font-medium">条件3：价格与均线</div><label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={cCond3.enabled} onChange={e => setCCond3((p: any) => ({...p, enabled: e.target.checked}))} disabled={computingC} /><span>启用</span></label></div>
          <div className="text-xs text-gray-500 mt-1">说明：选择用“收盘/最高/最低”与 MA(Y) 比较，要求在均线上方。</div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mt-2">
            <div>
              <div>用于比较的价格</div>
              <select className="mt-1 px-2 py-1 border rounded w-full" value={cCond3.price_field} onChange={e => setCCond3((p: any) => ({...p, price_field: e.target.value as any}))} disabled={computingC}>
                <option value="close">收盘价</option>
                <option value="high">最高价</option>
                <option value="low">最低价</option>
              </select>
            </div>
            <div>
              <div>均线天数（默认 60）</div>
              <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={cCond3.ma_days} onChange={e => setCCond3((p: any) => ({...p, ma_days: parseInt(e.target.value||'60',10)}))} disabled={computingC} />
            </div>
            <div>
              <div>比较关系</div>
              <select className="mt-1 px-2 py-1 border rounded w-full" value={cCond3.relation} onChange={e => setCCond3((p: any) => ({...p, relation: e.target.value as any}))} disabled={computingC}>
                <option value="ge">≥</option>
                <option value="gt">＞</option>
              </select>
            </div>
          </div>
        </div>

        {actions}
      </div>
    </div>
  );
}
