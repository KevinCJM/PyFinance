//
// eslint-disable-next-line @typescript-eslint/no-unused-vars

export default function BParamsPanel({
  bCond1, setBCond1,
  bCond2MA, setBCond2MA,
  bCond2, setBCond2,
  bCond4VR, setBCond4VR,
  bCond3, setBCond3,
  bCond4, setBCond4,
  computingB,
  actions,
}: any) {
  return (
    <div className="mt-4 bg-white p-4 rounded shadow">
      <div className="font-semibold mb-2">寻找B点（基于 A→B 条件组合）</div>
      <div className="mt-3 space-y-4">
        {/* 条件1：时间要求 */}
        <div className="border rounded p-3">
          <div className="flex items-center justify-between">
            <div className="font-medium">条件1：时间要求</div>
            <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond1.enabled} onChange={e => setBCond1((p: any) => ({...p, enabled: e.target.checked}))} disabled={computingB} /><span>启用</span></label>
          </div>
          <div className="mt-1 text-xs text-gray-500">
            说明：从 A 点出现到今天，至少要间隔一定天数；可选上限天数。并可设置是否允许一个 A 点对应多个 B 点（默认允许）。
          </div>
          <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <div>
              <div>从 A 点到今天至少经过（天）</div>
              <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond1.min_days_from_a} onChange={e => setBCond1((p: any) => ({...p, min_days_from_a: parseInt(e.target.value||'60',10)}))} disabled={computingB} />
            </div>
            <div>
              <div>从 A 点到今天不超过（天，可留空）</div>
              <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond1.max_days_from_a as any}
                     onChange={e => setBCond1((p: any) => ({...p, max_days_from_a: e.target.value === '' ? '' : parseInt(e.target.value,10)}))}
                     disabled={computingB} placeholder="留空表示不限制" />
            </div>
            <div className="flex items-center gap-2 col-span-2">
              <input type="checkbox" className="mr-2" checked={bCond1.allow_multi_b_per_a} onChange={e => setBCond1((p: any) => ({...p, allow_multi_b_per_a: e.target.checked}))} disabled={computingB} />
              <span>是否允许一个A点可以对应多个B点</span>
            </div>
          </div>
        </div>

        {/* 条件2：短期线在长期线之上 */}
        <div className="border rounded p-3">
          <div className="flex items-center justify-between">
            <div className="font-medium">条件2：短期线在长期线之上</div>
            <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond2MA.enabled} onChange={e => setBCond2MA((p: any) => ({...p, enabled: e.target.checked}))} disabled={computingB} /><span>启用</span></label>
          </div>
          <div className="mt-1 text-xs text-gray-500">说明：要求“短期平均线”（如 MA5）在“长期平均线”（如 MA60）之上。</div>
          {/* 模块一：均线参数 */}
          <div className="mt-2 border rounded p-3">
            <div className="text-sm font-medium mb-2">均线参数</div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div>
                <div>短期窗口</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond2MA.above_maN_window} onChange={e => setBCond2MA((p: any) => ({...p, above_maN_window: parseInt(e.target.value||'5',10)}))} disabled={computingB} />
              </div>
              <div>
                <div>长期窗口</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond2MA.long_ma_days} onChange={e => setBCond2MA((p: any) => ({...p, long_ma_days: parseInt(e.target.value||'60',10)}))} disabled={computingB} />
              </div>
              <div>
                <div>在上方天数（累计）</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond2MA.above_maN_days} onChange={e => setBCond2MA((p: any) => ({...p, above_maN_days: parseInt(e.target.value||'15',10)}))} disabled={computingB} />
              </div>
              <div>
                <div>在上方比例%</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond2MA.above_maN_ratio as any} onChange={e => setBCond2MA((p: any) => ({...p, above_maN_ratio: parseInt(e.target.value||'60',10)}))} disabled={computingB} />
              </div>
              <div className="flex items-center gap-2 col-span-2">
                <input type="checkbox" className="mr-2" checked={bCond2MA.above_maN_consecutive} onChange={e => setBCond2MA((p: any) => ({...p, above_maN_consecutive: e.target.checked}))} disabled={computingB} />
                <span>“在上方天数”是否要求连续（默认否）</span>
              </div>
              <div className="flex items-center gap-2 col-span-2">
                <input type="checkbox" className="mr-2" checked={bCond2MA.max_maN_below_days > 0} onChange={e => setBCond2MA((p: any) => ({...p, max_maN_below_days: e.target.checked ? 5 : 0}))} disabled={computingB} />
                <span>容忍短期小幅跌破（最大天数）</span>
              </div>
            </div>
          </div>
        </div>

        {/* 条件3：接近长期线 + 阴线/收≤昨收 */}
        <div className="border rounded p-3">
          <div className="flex items-center justify之间">
            <div className="font-medium">条件3：接近长期线 + 阴线/收≤昨收</div>
            <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond2.enabled} onChange={e => setBCond2((p: any) => ({...p, enabled: e.target.checked}))} disabled={computingB} /><span>启用</span></label>
          </div>
          <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <div>
              <div>比较价格</div>
              <select className="mt-1 px-2 py-1 border rounded w-full" value={bCond2.touch_price} onChange={e => setBCond2((p: any) => ({...p, touch_price: e.target.value as any}))} disabled={computingB}>
                <option value="low">最低价</option>
                <option value="close">收盘价</option>
              </select>
            </div>
            <div>
              <div>比较关系</div>
              <select className="mt-1 px-2 py-1 border rounded w-full" value={bCond2.touch_relation} onChange={e => setBCond2((p: any) => ({...p, touch_relation: e.target.value as any}))} disabled={computingB}>
                <option value="le">≤</option>
                <option value="lt">＜</option>
              </select>
            </div>
            <div className="flex items-center gap-2"><input type="checkbox" checked={bCond2.require_bearish} onChange={e => setBCond2((p: any) => ({...p, require_bearish: e.target.checked}))} disabled={computingB} /><span>要求阴线</span></div>
            <div className="flex items-center gap-2"><input type="checkbox" checked={bCond2.require_close_le_prev} onChange={e => setBCond2((p: any) => ({...p, require_close_le_prev: e.target.checked}))} disabled={computingB} /><span>收≤昨收</span></div>
          </div>
        </div>

        {/* 条件4：量能上限（VR1） */}
        <div className="border rounded p-3">
          <div className="flex items中心 justify-between">
            <div className="font-medium">条件4：量能上限（VR1）</div>
            <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond4VR.enabled} onChange={e => setBCond4VR((p: any) => ({...p, enabled: e.target.checked}))} disabled={computingB} /><span>启用</span></label>
          </div>
          <div className="mt-2 grid grid-cols-2 gap-3 text-sm">
            <div><div>VR1上限</div><input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond4VR.vr1_max as any} onChange={e => setBCond4VR((p: any) => ({...p, vr1_max: e.target.value === '' ? '' : parseFloat(e.target.value)}))} disabled={computingB || !bCond4VR.enabled} /></div>
            <div><div>回看天数</div><input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond4VR.recent_max_vol_window} onChange={e => setBCond4VR((p: any) => ({...p, recent_max_vol_window: parseInt(e.target.value||'10',10)}))} disabled={computingB || !bCond4VR.enabled} /></div>
          </div>
        </div>

        {/* 条件5：缩量（三个子模块） */}
        <div className="border rounded p-3">
          <div className="flex items-center justify-between">
            <div className="font-medium">条件5：缩量</div>
            <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond3.enabled} onChange={e => setBCond3((p: any) => ({...p, enabled: e.target.checked}))} disabled={computingB} /><span>启用</span></label>
          </div>
          {/* 子模块1：非放量（VR1） */}
          <div className="mt-2 border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">子模块1：非放量（VR1）</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={(bCond3 as any).vr1_enabled ?? false} onChange={e => setBCond3((p: any) => ({...p, vr1_enabled: e.target.checked}))} disabled={computingB} /><span>启用</span></label>
            </div>
            <div className="mt-1 text-xs text-gray-500">说明：与 C 点“放量VR1”相反，这里要求 VR1 ≤ 阈值（今天量不超过近N日最大量的若干倍）。</div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div>
                <div>参照天数 N（默认 10）</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={(bCond3 as any).recent_n ?? 10} onChange={e => setBCond3((p: any) => ({...p, recent_n: parseInt(e.target.value||'10',10)}))} disabled={computingB || !(bCond3 as any).vr1_enabled} />
              </div>
              <div>
                <div>VR1 阈值倍数（默认 1.2）</div>
                <input type="number" step="0.01" className="mt-1 px-2 py-1 border rounded w-full" value={(bCond3 as any).vr1_max ?? 1.2} onChange={e => setBCond3((p: any) => ({...p, vr1_max: parseFloat(e.target.value||'1.2')}))} disabled={computingB || !(bCond3 as any).vr1_enabled} />
              </div>
            </div>
          </div>
          {/* 子模块2：量均比较（短≤长） */}
          <div className="mt-2 border rounded p-3">
            <div className="flex items-center justify-between"><div className="text-sm font-medium">子模块2：量均比较（短≤长）</div><label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={(bCond3 as any).vma_rel_enabled ?? false} onChange={e => setBCond3((p: any) => ({...p, vma_rel_enabled: e.target.checked}))} disabled={computingB || !bCond3.enabled} /><span>启用</span></label></div>
            <div className="mt-1 text-xs text-gray-500">说明：与 C 点“短量均&gt;长量均”相反，这里要求短期量均≤长期量均。</div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div>
                <div>短天数（默认 5）</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond3.short_days} onChange={e => setBCond3((p: any) => ({...p, short_days: parseInt(e.target.value||'5',10)}))} disabled={computingB || !bCond3.enabled || !(bCond3 as any).vma_rel_enabled} />
              </div>
              <div>
                <div>长天数（默认 10）</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond3.long_days} onChange={e => setBCond3((p: any) => ({...p, long_days: parseInt(e.target.value||'10',10)}))} disabled={computingB || !bCond3.enabled || !(bCond3 as any).vma_rel_enabled} />
              </div>
            </div>
          </div>
          {/* 子模块3：近X日量连降（严格递减） */}
          <div className="mt-2 border rounded p-3">
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium">子模块3：近X日量连降</div>
              <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={(bCond3 as any).vol_down_enabled ?? false} onChange={e => setBCond3((p: any) => ({...p, vol_down_enabled: e.target.checked}))} disabled={computingB || !bCond3.enabled} /><span>启用</span></label>
            </div>
            <div className="mt-1 text-xs text-gray-500">说明：要求最近X日（含当日）成交量严格递减，体现缩量走稳。</div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div>
                <div>X（日）</div>
                <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={(bCond3 as any).vol_decreasing_days ?? 3} onChange={e => setBCond3((p: any) => ({...p, vol_decreasing_days: parseInt(e.target.value||'3',10)}))} disabled={computingB || !bCond3.enabled || !(bCond3 as any).vol_down_enabled} />
              </div>
            </div>
          </div>
        </div>

        {/* 条件6：价稳 */}
        <div className="border rounded p-3">
          <div className="flex items-center justify-between">
            <div className="font-medium">条件6：价稳</div>
            <label className="inline-flex items-center space-x-2 text-sm"><input type="checkbox" checked={bCond4.enabled} onChange={e => setBCond4((p: any) => ({...p, enabled: e.target.checked}))} disabled={computingB} /><span>启用</span></label>
          </div>
          <div className="mt-1 text-xs text-gray-500">
            说明：希望“今天的最低价”相对这段时间的“前期最低价”比较稳定。
            你可以选三种方式：
          </div>
          <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <div>
              <div>模式</div>
              <select className="mt-1 px-2 py-1 border rounded w-full" value={bCond4.price_stable_mode} onChange={e => setBCond4((p: any) => ({...p, price_stable_mode: e.target.value as any}))} disabled={computingB}>
                <option value="no_new_low">不创新低</option>
                <option value="ratio">最大跌幅</option>
                <option value="atr">ATR</option>
              </select>
            </div>
            <div>
              <div>最大跌幅</div>
              <input type="number" step="0.001" className="mt-1 px-2 py-1 border rounded w-full" value={bCond4.max_drop_ratio} onChange={e => setBCond4((p: any) => ({...p, max_drop_ratio: parseFloat(e.target.value||'0.03')}))} disabled={computingB || bCond4.price_stable_mode!=='ratio'} />
            </div>
            <div>
              <div>参考过去多少天的波动</div>
              <input type="number" className="mt-1 px-2 py-1 border rounded w-full" value={bCond4.use_atr_window} onChange={e => setBCond4((p: any) => ({...p, use_atr_window: parseInt(e.target.value||'14',10)}))} disabled={computingB || bCond4.price_stable_mode!=='atr'} />
            </div>
            <div>
              <div>给多少倍的“波动缓冲”</div>
              <input type="number" step="0.01" className="mt-1 px-2 py-1 border rounded w-full" value={bCond4.atr_buffer} onChange={e => setBCond4((p: any) => ({...p, atr_buffer: parseFloat(e.target.value||'0.5')}))} disabled={computingB || bCond4.price_stable_mode!=='atr'} />
            </div>
          </div>
        </div>

        {actions}
      </div>
    </div>
  );
}
