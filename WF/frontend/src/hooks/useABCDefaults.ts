import { useMemo } from 'react'

// A 点参数类型
export type ACond1 = { 启用: boolean; 长均线窗口: number; 下跌跨度: number }
export type ACond2 = { 启用: boolean; 短均线集合: string; 长均线窗口: number; 上穿完备窗口: number; 必须满足的短均线: string; 全部满足: boolean }
export type ACond3 = { 启用: boolean; 确认回看天数: number; 确认均线窗口: string; 确认价格列: 'open'|'high'|'low'|'close' }
export type ACondVol = { 启用: boolean; vr1_enabled: boolean; 对比天数: number; 倍数: number; vma_cmp_enabled: boolean; 短期天数: number; 长期天数: number; vol_up_enabled: boolean; 量连升天数: number }

// B 点参数类型（与页面/面板结构对齐）
export type BCond1 = { enabled: boolean; min_days_from_a: number; max_days_from_a: number|''; allow_multi_b_per_a: boolean }
export type BCond2MA = { enabled: boolean; mode?: 'ratio'|'days'; above_maN_window: number; above_maN_days: number; above_maN_consecutive: boolean; max_maN_below_days: number; long_ma_days: number; above_maN_ratio?: number }
export type BCond2 = { enabled: boolean; touch_price: 'low'|'close'; touch_relation: 'le'|'lt'; require_bearish: boolean; require_close_le_prev: boolean; long_ma_days: number }
export type BCond4VR = { enabled: boolean; vr1_max: number | ''; recent_max_vol_window: number }
export type BCond3 = { enabled: boolean; dryness_ratio_max: number; require_vol_le_vma10: boolean; dryness_recent_window: number; dryness_recent_min_days: number; short_days: number; long_days: number; vol_compare_long_window: number; vr1_enabled?: boolean; vma_rel_enabled?: boolean; vol_down_enabled?: boolean; vol_decreasing_days?: number }
export type BCond4 = { enabled: boolean; price_stable_mode: 'no_new_low'|'ratio'|'atr'; max_drop_ratio: number; use_atr_window: number; atr_buffer: number }

// C 点参数类型
export type CCond1 = { enabled: boolean; max_days_from_b: number }
export type CCond2 = { enabled: boolean; vr1_enabled: boolean; recent_n: number; vol_multiple: number; vma_cmp_enabled: boolean; vma_short_days: number; vma_long_days: number; vol_up_enabled: boolean; vol_increasing_days: number }
export type CCond3 = { enabled: boolean; price_field: 'close'|'high'|'low'; ma_days: number; relation: 'ge'|'gt' }

export function useABCDefaults() {
  const aDefaults = useMemo(() => ({
    aCond1: { 启用: true, 长均线窗口: 60, 下跌跨度: 30 } as ACond1,
    aCond2: { 启用: true, 短均线集合: '5,10', 长均线窗口: 60, 上穿完备窗口: 3, 必须满足的短均线: '', 全部满足: false } as ACond2,
    aCond3: { 启用: false, 确认回看天数: 0, 确认均线窗口: '', 确认价格列: 'high' } as ACond3,
    aCondVol: { 启用: true, vr1_enabled: false, 对比天数: 10, 倍数: 2.0, vma_cmp_enabled: true, 短期天数: 5, 长期天数: 10, vol_up_enabled: true, 量连升天数: 3 } as ACondVol,
  }), []);

  const bDefaults = useMemo(() => ({
    bCond1: { enabled: true, min_days_from_a: 60, max_days_from_a: '' as number|'', allow_multi_b_per_a: true } as BCond1,
    bCond2MA: { enabled: true, mode: 'ratio', above_maN_window: 5, above_maN_days: 15, above_maN_consecutive: false, max_maN_below_days: 5, long_ma_days: 60, above_maN_ratio: 60 } as BCond2MA,
    bCond2: { enabled: true, touch_price: 'low', touch_relation: 'le', require_bearish: false, require_close_le_prev: false, long_ma_days: 60 } as BCond2,
    bCond4VR: { enabled: false, vr1_max: '' as number|'', recent_max_vol_window: 10 } as BCond4VR,
    bCond3: { enabled: true, dryness_ratio_max: 0.8, require_vol_le_vma10: true, dryness_recent_window: 0, dryness_recent_min_days: 0, short_days: 5, long_days: 10, vol_compare_long_window: 10, vr1_enabled: false, vma_rel_enabled: true, vol_down_enabled: true, vol_decreasing_days: 3 } as BCond3,
    bCond4: { enabled: false, price_stable_mode: 'no_new_low', max_drop_ratio: 0.03, use_atr_window: 14, atr_buffer: 0.5 } as BCond4,
  }), []);

  const cDefaults = useMemo(() => ({
    cCond1: { enabled: true, max_days_from_b: 60 } as CCond1,
    cCond2: { enabled: true, vr1_enabled: false, recent_n: 10, vol_multiple: 2.0, vma_cmp_enabled: false, vma_short_days: 5, vma_long_days: 10, vol_up_enabled: true, vol_increasing_days: 3 } as CCond2,
    cCond3: { enabled: true, price_field: 'close', ma_days: 60, relation: 'ge' } as CCond3,
  }), []);

  return { aDefaults, bDefaults, cDefaults };
}
