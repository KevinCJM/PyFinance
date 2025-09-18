# -*- encoding: utf-8 -*-
"""
@File: T01_etf_cluster.py
@Modify Time: 2025/9/18 14:30
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
    """
    对ETF收益率数据进行预处理，为聚类分析做准备

    参数:
        etf_returns (DataFrame): ETF收益率数据，行表示时间，列表示不同的ETF

    返回:
        DataFrame: 转置后的ETF收益率数据，缺失值已填充为0
    """
    # 转置数据框，使ETF变为行，时间变为列
    etf_returns_T = etf_returns.T

    # 填充缺失值为0，避免聚类算法处理缺失数据时出错
    etf_returns_T = etf_returns_T.fillna(0)

    return etf_returns_T


def calculate_optimal_k_metrics(data, max_k=20):
    """
    计算不同K值下的聚类评估指标，用于确定最优K值

    参数:
        data: array-like, 聚类分析的输入数据
        max_k: int, 最大聚类数，默认为20

    返回:
        k_range: range对象, K值范围(2到max_k)
        inertias: list, 每个K值对应的簇内平方和(inertia)
        silhouette_scores: list, 每个K值对应的轮廓系数
    """
    inertias, silhouette_scores = [], []
    k_range = range(2, max_k + 1)
    print(f"正在计算最优K值 (K从2到{max_k})...")

    # 遍历不同的K值，计算对应的评估指标
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    return k_range, inertias, silhouette_scores


def run_kmeans(data, n_clusters):
    """
    执行K-means聚类算法

    参数:
        data: array-like, 聚类分析的输入数据
        n_clusters: int, 聚类数量

    返回:
        results_df: DataFrame, 包含股票代码和对应聚类标签的数据框
        kmeans: KMeans对象, 训练好的K-means模型
    """
    print(f"正在执行K-means聚类 (K={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(data)

    # 构建结果数据框，包含股票代码和聚类标签
    results_df = pd.DataFrame({'ts_code': data.index, 'cluster': labels})

    return results_df, kmeans


def find_representative_etfs(cluster_data, kmeans_results, kmeans_model, n_representatives=3):
    """
    从K-means聚类结果中为每个簇找到最具代表性的ETF基金

    参数:
        cluster_data: DataFrame, 包含ETF特征数据的DataFrame，索引为ts_code
        kmeans_results: DataFrame, K-means聚类结果，包含'cluster'列和'ts_code'列
        kmeans_model: KMeans模型对象, 已训练好的K-means聚类模型
        n_representatives: int, 每个簇选取的代表性ETF数量，默认为3

    返回:
        tuple: (representatives, all_rep_codes)
            - representatives: dict, 键为簇ID，值为该簇代表性ETF的ts_code列表
            - all_rep_codes: list, 所有代表性ETF的ts_code列表
    """
    representatives, all_rep_codes = {}, []
    cluster_centers = kmeans_model.cluster_centers_

    # 遍历每个簇，找到最具代表性的ETF
    for cluster_id in range(kmeans_model.n_clusters):
        cluster_members_codes = kmeans_results[kmeans_results['cluster'] == cluster_id]['ts_code']
        if cluster_members_codes.empty: continue

        # 获取当前簇中所有成员的特征数据
        cluster_member_data = cluster_data.loc[cluster_members_codes]
        cluster_center = cluster_centers[cluster_id].reshape(1, -1)

        # 计算每个成员到簇中心的欧氏距离
        distances = cdist(cluster_member_data, cluster_center, 'euclidean')
        distance_df = pd.DataFrame({'ts_code': cluster_member_data.index, 'distance': distances.flatten()})

        # 选择距离最近的n个ETF作为代表
        closest_etfs = distance_df.sort_values(by='distance').head(n_representatives)
        rep_codes = closest_etfs['ts_code'].tolist()
        representatives[cluster_id] = rep_codes
        all_rep_codes.extend(rep_codes)

    return representatives, all_rep_codes


def calculate_summary_statistics(etf_return_pivot):
    """
    计算ETF收益率的摘要统计信息

    参数:
        etf_return_pivot (pandas.DataFrame): ETF收益率数据的透视表，行表示时间，列表示不同的ETF

    返回:
        pandas.DataFrame: 包含每个ETF的年化收益率、年化波动率和夏普比率的统计摘要
    """
    print("正在计算摘要统计信息...")
    trading_days = 252
    # 计算年化收益率、年化波动率和夏普比率
    annualized_return = etf_return_pivot.mean() * trading_days
    annualized_volatility = etf_return_pivot.std() * np.sqrt(trading_days)
    sharpe_ratio = annualized_return / annualized_volatility
    return pd.DataFrame({'annualized_return': annualized_return, 'annualized_volatility': annualized_volatility,
                         'sharpe_ratio': sharpe_ratio})


def generate_cluster_report(cluster_data, kmeans_results, etf_return_pivot, name_map, k_range, inertias,
                            silhouette_scores, representative_codes, output_dir='data'):
    """
    生成ETF聚类分析的可视化报告，包括主分析图和分层聚类树状图。

    参数:
        cluster_data (pd.DataFrame): 用于聚类的数据，每行代表一个ETF，列为特征。
        kmeans_results (pd.DataFrame): K-means聚类结果，包含'ts_code'和'cluster'列。
        etf_return_pivot (pd.DataFrame): ETF收益率的透视表，列为ETF代码，行为时间。
        name_map (dict): ETF代码到名称的映射字典。
        k_range (range or list): 用于肘部法则和轮廓系数分析的K值范围。
        inertias (list): 每个K值对应的簇内平方和（Inertia）。
        silhouette_scores (list): 每个K值对应的轮廓系数。
        representative_codes (list): 代表性ETF的代码列表。
        output_dir (str): 报告输出目录，默认为'data'。

    返回值:
        无返回值。生成HTML格式的可视化报告并保存至指定目录。
    """
    print("正在生成可视化报告...")
    n_clusters = kmeans_results['cluster'].nunique()

    # --- Part 1: Main Analysis Figure ---
    # 使用PCA将聚类数据降维至二维，便于可视化
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(cluster_data)
    pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=cluster_data.index)
    pca_df = pca_df.merge(kmeans_results, left_index=True, right_on='ts_code').set_index('ts_code')
    pca_df['name'] = pca_df.index.map(name_map)
    pca_df['is_representative'] = pca_df.index.isin(representative_codes)

    # 统计每个聚类的ETF数量
    cluster_sizes = kmeans_results.groupby('cluster').size().reset_index(name='count')

    # 计算代表性ETF之间的相关系数矩阵
    rep_etf_returns = etf_return_pivot[representative_codes].fillna(0)
    correlation_matrix = rep_etf_returns.corr()

    # 构建热力图标签，包含ETF名称和所属聚类类别
    code_to_cluster_map = kmeans_results.set_index('ts_code')['cluster'].to_dict()
    heatmap_labels = []
    for code in correlation_matrix.columns:
        name = name_map.get(code, code)
        cluster_id = code_to_cluster_map.get(code, 'N/A')
        heatmap_labels.append(f"{name} (类别 {cluster_id})")

    # 创建主分析图，包含肘部法则、轮廓系数、PCA聚类图、聚类数量柱状图和相关系数热力图
    fig_main = make_subplots(
        rows=3, cols=2,
        specs=[[{}, {}], [{}, {}], [{'colspan': 2}, None]],
        subplot_titles=('肘部法则', '轮廓系数', 'K-means聚类(PCA)', '各类别ETF数量', '典型ETF相关系数矩阵'),
        vertical_spacing=0.15
    )

    fig_main.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers', name='Inertia'), row=1, col=1)
    fig_main.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers', name='Silhouette Score'),
                       row=1, col=2)

    # 绘制PCA聚类图，区分代表性和非代表性ETF
    for cluster_id in range(n_clusters):
        cluster_df = pca_df[pca_df['cluster'] == cluster_id]
        reps_df, non_reps_df = cluster_df[cluster_df['is_representative']], cluster_df[~cluster_df['is_representative']]
        fig_main.add_trace(
            go.Scatter(x=non_reps_df['PC1'], y=non_reps_df['PC2'], mode='markers', name=f'C {cluster_id}',
                       text=non_reps_df['name'], hoverinfo='text+name', marker={'size': 8}), row=2, col=1)
        fig_main.add_trace(go.Scatter(x=reps_df['PC1'], y=reps_df['PC2'], mode='markers', name=f'Rep C {cluster_id}',
                                      text=reps_df['name'], hoverinfo='text+name',
                                      marker={'symbol': 'star', 'size': 14, 'line': {'width': 1, 'color': 'black'}}),
                           row=2, col=1)

    fig_main.add_trace(go.Bar(x=cluster_sizes['cluster'], y=cluster_sizes['count'], name='ETF Count'), row=2, col=2)

    fig_main.add_trace(
        go.Heatmap(z=correlation_matrix.values, x=heatmap_labels, y=heatmap_labels, colorscale='RdBu', zmin=-1, zmax=1,
                   showscale=False), row=3, col=1)

    fig_main.update_layout(height=1400, width=1200, title_text="ETF 聚类分析报告 - 主分析", showlegend=False)
    report_path = os.path.join(output_dir, 'cluster_analysis_report.html')
    fig_main.write_html(report_path)

    # --- Part 2: Dendrograms Figure ---
    if n_clusters > 0:
        # 为每个聚类生成分层聚类树状图
        dendro_rows = (n_clusters + 1) // 2
        fig_dendro = make_subplots(
            rows=dendro_rows, cols=2,
            subplot_titles=tuple(f'分层聚类 - 类别{i}' for i in range(n_clusters))
        )
        for cluster_id in range(n_clusters):
            row, col = 1 + cluster_id // 2, 1 + cluster_id % 2
            cluster_etfs = kmeans_results[kmeans_results['cluster'] == cluster_id]['ts_code']
            if len(cluster_etfs) > 1:
                cluster_return_data = etf_return_pivot[cluster_etfs].T.fillna(0)
                dendro_labels = [name_map.get(code, code) for code in cluster_return_data.index]
                dendro_fig = ff.create_dendrogram(cluster_return_data, labels=dendro_labels, color_threshold=0.8)
                for trace in dendro_fig['data']:
                    fig_dendro.add_trace(trace, row=row, col=col)
                leaf_x, leaf_labels = dendro_fig['layout']['xaxis']['tickvals'], dendro_fig['layout']['xaxis'][
                    'ticktext']
                fig_dendro.add_trace(go.Scatter(x=leaf_x, y=[0] * len(leaf_x), mode='markers',
                                                marker={'color': 'rgba(0,0,0,0)', 'size': 15}, text=leaf_labels,
                                                hoverinfo='text'), row=row, col=col)
                fig_dendro.update_xaxes(showticklabels=False, row=row, col=col)

        fig_dendro.update_layout(height=400 * dendro_rows, width=1200, title_text="分层聚类树状图", showlegend=False)
        with open(report_path, 'a') as f:
            f.write(fig_dendro.to_html(full_html=False, include_plotlyjs=False))

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
        print(f"错误: {e}")
        exit()

    print("\n步骤 2: 计算最优K值指标...")
    k_range, inertias, silhouette_scores = calculate_optimal_k_metrics(cluster_data, max_k=MAX_K_TO_TEST)

    print(f"\n步骤 3: 执行K-means聚类 (K={OPTIMAL_K})...")
    kmeans_results, kmeans_model = run_kmeans(cluster_data, n_clusters=OPTIMAL_K)
    kmeans_results['name'] = kmeans_results['ts_code'].map(name_map)

    print("\n步骤 4: 找出每个类别的典型ETF...")
    rep_etfs_by_cluster, all_rep_codes = find_representative_etfs(
        cluster_data, kmeans_results, kmeans_model, n_representatives=N_REPRESENTATIVES)
    all_rep_codes = sorted(list(set(all_rep_codes)))
    print("各类别典型ETF如下:")
    for cluster_id, etf_codes in rep_etfs_by_cluster.items():
        etf_names = [name_map.get(c, c) for c in etf_codes]
        print(f"  类别 {cluster_id}: {', '.join(etf_names)}")

    print("\n步骤 5: 计算摘要统计信息并合并结果...")
    summary_stats = calculate_summary_statistics(etf_return_pivot)
    final_results = kmeans_results.merge(summary_stats, left_on='ts_code', right_index=True)
    final_results['is_representative'] = final_results['ts_code'].isin(all_rep_codes)
    final_results = final_results.sort_values(by=['cluster', 'is_representative', 'ts_code'],
                                              ascending=[True, False, True])
    final_results.rename(
        columns={'cluster': '类别代码', 'ts_code': 'ETF代码', 'name': 'ETF名称', 'annualized_return': '年化收益率',
                 'annualized_volatility': '年化波动率', 'sharpe_ratio': '夏普比率', 'is_representative': '是否典型'},
        inplace=True)
    final_results = final_results[
        ['类别代码', 'ETF代码', 'ETF名称', '是否典型', '年化收益率', '年化波动率', '夏普比率']]

    print("\n步骤 6: 保存最终结果到 Excel, Parquet 和 CSV 文件...")
    excel_path, = os.path.join(OUTPUT_DIR, 'cluster_results.xlsx')
    parquet_path = os.path.join(OUTPUT_DIR, 'cluster_results.parquet')
    csv_path = os.path.join(OUTPUT_DIR, 'cluster_results.csv')
    final_results.to_excel(excel_path, index=False)
    final_results.to_parquet(parquet_path, index=False)
    final_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"结果已保存至: {excel_path}, {parquet_path}, {csv_path}")

    print("\n步骤 7: 生成可视化报告...")
    generate_cluster_report(cluster_data, kmeans_results, etf_return_pivot,
                            name_map, k_range, inertias,
                            silhouette_scores, all_rep_codes, output_dir=OUTPUT_DIR)

    print("\n\n分析完成！")
