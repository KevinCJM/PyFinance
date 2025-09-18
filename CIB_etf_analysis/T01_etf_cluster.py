# -*- encoding: utf-8 -*-
"""
@File: T01_etf_cluster.py
@Modify Time: 2025/9/18 12:00
@Author: Kevin-Chen
@Descriptions: ETF聚类分析 (K-means + 分层聚类) 与 Plotly 可视化
"""
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff


def read_data(nv_parquet, trade_day_parquet):
    """
    读取并预处理ETF净值数据和交易日历数据，计算ETF日收益率并返回透视后的结果以及名称映射。
    :return: (pandas.DataFrame, dict) -> (透视表, ts_code到name的映射)
    """
    if not os.path.exists(nv_parquet) or not os.path.exists(trade_day_parquet):
        raise FileNotFoundError(
            f"数据文件缺失，请确保 {nv_parquet} 和 {trade_day_parquet} 存在。"
            "您可能需要先运行 A01_get_data.py 来下载数据。"
        )
    etf_data = pd.read_parquet(nv_parquet)
    etf_data = etf_data[["ts_code", "name", "date", "adj_nav"]]
    etf_data['date'] = pd.to_datetime(etf_data['date'])
    etf_data = etf_data[etf_data['date'] >= '2015-01-01']
    etf_data = etf_data.sort_values(by=['ts_code', 'date'])
    
    name_map = etf_data[['ts_code', 'name']].drop_duplicates().set_index('ts_code')['name'].to_dict()

    trade_day_df = pd.read_parquet(trade_day_parquet)
    trade_day_df = trade_day_df[trade_day_df['is_open'] == 1]
    trade_day_df['cal_date'] = pd.to_datetime(trade_day_df['cal_date'])
    etf_data = etf_data[etf_data['date'].isin(trade_day_df['cal_date'])]
    print(f"ETF数量: {etf_data['ts_code'].nunique()}")
    min_count = etf_data['date'].nunique() * 0.5
    valid_etfs = etf_data.groupby('ts_code').filter(lambda x: len(x) >= min_count)['ts_code'].unique()
    etf_data = etf_data[etf_data['ts_code'].isin(valid_etfs)]
    print(f"剩余ETF数量: {etf_data['ts_code'].nunique()}")
    latest_date = etf_data['date'].max()
    recent_threshold = latest_date - pd.Timedelta(days=20)
    recent_etfs = etf_data[etf_data['date'] >= recent_threshold]['ts_code'].unique()
    etf_data = etf_data[etf_data['ts_code'].isin(recent_etfs)]
    print(f"剩余ETF数量: {etf_data['ts_code'].nunique()}")
    etf_data['return'] = etf_data.groupby('ts_code')['adj_nav'].pct_change(fill_method=None)
    etf_returns = etf_data.pivot(index='date', columns='ts_code', values='return')
    return etf_returns, name_map


def preprocess_for_clustering(etf_returns):
    etf_returns_T = etf_returns.T
    etf_returns_T = etf_returns_T.fillna(0)
    return etf_returns_T


def calculate_optimal_k_metrics(data, max_k=20):
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    print(f"正在计算最优K值 (K从2到{max_k})...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    return k_range, inertias, silhouette_scores


def run_kmeans(data, n_clusters):
    print(f"正在执行K-means聚类 (K={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(data)
    results_df = pd.DataFrame({'ts_code': data.index, 'cluster': labels})
    return results_df, kmeans


def find_representative_etfs(cluster_data, kmeans_results, kmeans_model, n_representatives=3):
    """
    为每个类别找到最具代表性的ETF。
    :return: (dict, list) -> ({类别: [ETF代码列表]}, 所有代表性ETF代码的扁平列表)
    """
    representatives = {}
    all_rep_codes = []
    cluster_centers = kmeans_model.cluster_centers_
    
    for cluster_id in range(kmeans_model.n_clusters):
        cluster_members_codes = kmeans_results[kmeans_results['cluster'] == cluster_id]['ts_code']
        if cluster_members_codes.empty:
            continue
        cluster_member_data = cluster_data.loc[cluster_members_codes]
        
        cluster_center = cluster_centers[cluster_id].reshape(1, -1)
        
        distances = cdist(cluster_member_data, cluster_center, 'euclidean')
        
        distance_df = pd.DataFrame({
            'ts_code': cluster_member_data.index,
            'distance': distances.flatten()
        })
        
        closest_etfs = distance_df.sort_values(by='distance').head(n_representatives)
        rep_codes = closest_etfs['ts_code'].tolist()
        representatives[cluster_id] = rep_codes
        all_rep_codes.extend(rep_codes)
            
    return representatives, all_rep_codes


def calculate_summary_statistics(etf_return_pivot):
    """
    计算每个ETF的摘要统计信息（年化收益率、波动率、夏普比率）。
    """
    print("正在计算摘要统计信息...")
    trading_days = 252
    annualized_return = etf_return_pivot.mean() * trading_days
    annualized_volatility = etf_return_pivot.std() * np.sqrt(trading_days)
    sharpe_ratio = annualized_return / annualized_volatility
    
    summary_df = pd.DataFrame({
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sharpe_ratio': sharpe_ratio
    })
    return summary_df


def generate_cluster_report(cluster_data, kmeans_results, etf_return_pivot, name_map, k_range, inertias, silhouette_scores, representative_codes, output_dir='data'):
    """
    使用Plotly生成包含多个图表的聚类分析报告。
    """
    print("正在生成可视化报告...")
    n_clusters = kmeans_results['cluster'].nunique()

    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(cluster_data)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=cluster_data.index)
    pca_df = pca_df.merge(kmeans_results, left_index=True, right_on='ts_code').set_index('ts_code')
    pca_df['name'] = pca_df.index.map(name_map)
    pca_df['is_representative'] = pca_df.index.isin(representative_codes)

    cluster_sizes = kmeans_results.groupby('cluster').size().reset_index(name='count')

    dendro_rows = (n_clusters + 1) // 2
    row_heights = [0.2, 0.3] + [0.5 / dendro_rows] * dendro_rows
    specs = [[{'colspan': 1}, {'colspan': 1}], [{'colspan': 1}, {'colspan': 1}]] + \
            [[{'colspan': 1}, {'colspan': 1}] for _ in range(dendro_rows)]
    
    subplot_titles = ('肘部法则确定最优K值', '轮廓系数确定最优K值',
                      'K-means 聚类 (PCA二维展示)', '各类别ETF数量') + \
                     tuple(f'分层聚类树状图 - 类别 {i}' for i in range(n_clusters))

    fig = make_subplots(rows=2 + dendro_rows, cols=2, specs=specs, subplot_titles=subplot_titles, row_heights=row_heights)

    fig.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers', name='Inertia'), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers', name='Silhouette Score'), row=1, col=2)

    for cluster_id in range(n_clusters):
        cluster_df = pca_df[pca_df['cluster'] == cluster_id]
        reps_df = cluster_df[cluster_df['is_representative']]
        non_reps_df = cluster_df[~cluster_df['is_representative']]
        
        fig.add_trace(go.Scatter(x=non_reps_df['PC1'], y=non_reps_df['PC2'], mode='markers', name=f'Cluster {cluster_id}', text=non_reps_df['name'], hoverinfo='text+name', marker={'size': 8}), row=2, col=1)
        fig.add_trace(go.Scatter(x=reps_df['PC1'], y=reps_df['PC2'], mode='markers', name=f'Rep. Cluster {cluster_id}', text=reps_df['name'], hoverinfo='text+name', marker={'symbol': 'star', 'size': 14, 'line': {'width': 1, 'color': 'black'}}), row=2, col=1)

    fig.add_trace(go.Bar(x=cluster_sizes['cluster'], y=cluster_sizes['count'], name='ETF Count'), row=2, col=2)

    for cluster_id in range(n_clusters):
        row, col = 3 + cluster_id // 2, 1 + cluster_id % 2
        cluster_etfs = kmeans_results[kmeans_results['cluster'] == cluster_id]['ts_code']
        if len(cluster_etfs) > 1:
            cluster_return_data = etf_return_pivot[cluster_etfs].T.fillna(0)
            dendro_labels = [name_map.get(code, code) for code in cluster_return_data.index]
            dendro_fig = ff.create_dendrogram(cluster_return_data, labels=dendro_labels, color_threshold=0.8)
            for trace in dendro_fig['data']:
                fig.add_trace(trace, row=row, col=col)
            leaf_x, leaf_labels = dendro_fig['layout']['xaxis']['tickvals'], dendro_fig['layout']['xaxis']['ticktext']
            fig.add_trace(go.Scatter(x=leaf_x, y=[0]*len(leaf_x), mode='markers', marker={'color':'rgba(0,0,0,0)','size':15}, text=leaf_labels, hoverinfo='text'), row=row, col=col)
            fig.update_xaxes(showticklabels=False, row=row, col=col)

    fig.update_layout(height=500+400*dendro_rows, title_text="ETF 聚类分析报告", showlegend=False)
    fig.update_xaxes(title_text="聚类数量 (K)", row=1, col=1)
    fig.update_yaxes(title_text="惯性 (Inertia)", row=1, col=1)
    fig.update_xaxes(title_text="聚类数量 (K)", row=1, col=2)
    fig.update_yaxes(title_text="轮廓系数", row=1, col=2)
    fig.update_xaxes(title_text="主成分1", row=2, col=1)
    fig.update_yaxes(title_text="主成分2", row=2, col=1)
    fig.update_xaxes(title_text="类别", row=2, col=2)
    fig.update_yaxes(title_text="ETF 数量", row=2, col=2)

    report_path = os.path.join(output_dir, 'cluster_analysis_report.html')
    fig.write_html(report_path)
    print(f"\n聚类分析报告已保存至: {report_path}")


if __name__ == '__main__':
    THE_NV_PARQUET, THE_TRADE_DAY_PARQUET = "data/etf_daily_df.parquet", "data/trade_day_df.parquet"
    OUTPUT_DIR, OPTIMAL_K, MAX_K_TO_TEST, N_REPRESENTATIVES = "data", 12, 20, 3

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("步骤 1: 读取并预处理数据...")
    try:
        etf_return_pivot, name_map = read_data(THE_NV_PARQUET, THE_TRADE_DAY_PARQUET)
        cluster_data = preprocess_for_clustering(etf_return_pivot)
    except FileNotFoundError as e:
        print(f"错误: {e}"); exit()

    print("\n步骤 2: 计算最优K值指标...")
    k_range, inertias, silhouette_scores = calculate_optimal_k_metrics(cluster_data, max_k=MAX_K_TO_TEST)

    print(f"\n步骤 3: 执行K-means聚类 (K={OPTIMAL_K})...")
    kmeans_results, kmeans_model = run_kmeans(cluster_data, n_clusters=OPTIMAL_K)
    kmeans_results['name'] = kmeans_results['ts_code'].map(name_map)

    print("\n步骤 4: 找出每个类别的典型ETF...")
    rep_etfs_by_cluster, all_rep_codes = find_representative_etfs(cluster_data, kmeans_results, kmeans_model, n_representatives=N_REPRESENTATIVES)
    print("各类别典型ETF如下:")
    for cluster_id, etf_codes in rep_etfs_by_cluster.items():
        etf_names = [name_map.get(c, c) for c in etf_codes]
        print(f"  类别 {cluster_id}: {', '.join(etf_names)}")

    print("\n步骤 5: 计算摘要统计信息并合并结果...")
    summary_stats = calculate_summary_statistics(etf_return_pivot)
    final_results = kmeans_results.merge(summary_stats, left_on='ts_code', right_index=True)
    final_results['is_representative'] = final_results['ts_code'].isin(all_rep_codes)
    final_results = final_results.sort_values(by=['cluster', 'is_representative', 'ts_code'], ascending=[True, False, True])
    final_results.rename(columns={'cluster':'类别代码','ts_code':'ETF代码','name':'ETF名称','annualized_return':'年化收益率','annualized_volatility':'年化波动率','sharpe_ratio':'夏普比率','is_representative':'是否典型'}, inplace=True)
    final_results = final_results[['类别代码', 'ETF代码', 'ETF名称', '是否典型', '年化收益率', '年化波动率', '夏普比率']]

    print("\n步骤 6: 保存最终结果到 Excel, Parquet 和 CSV 文件...")
    excel_path, parquet_path, csv_path = os.path.join(OUTPUT_DIR, 'cluster_results.xlsx'), os.path.join(OUTPUT_DIR, 'cluster_results.parquet'), os.path.join(OUTPUT_DIR, 'cluster_results.csv')
    final_results.to_excel(excel_path, index=False)
    final_results.to_parquet(parquet_path, index=False)
    final_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"结果已保存至: {excel_path}, {parquet_path}, {csv_path}")

    print("\n步骤 7: 生成可视化报告...")
    generate_cluster_report(cluster_data, kmeans_results, etf_return_pivot, name_map, k_range, inertias, silhouette_scores, all_rep_codes, output_dir=OUTPUT_DIR)

    print("\n\n分析完成！")
