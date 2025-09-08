document.addEventListener('DOMContentLoaded', () => {
    // --- DATA LOADING ---
    const fetchData = async () => {
        try {
            const [
                equityBhb, equityBf,
                assetBhb, assetBf,
                campisi
            ] = await Promise.all([
                fetch('../output/A_bhb_fof_agg.json').then(res => res.ok ? res.json() : Promise.resolve(null)),
                fetch('../output/A_bf_fof_agg.json').then(res => res.ok ? res.json() : Promise.resolve(null)),
                fetch('../output/B_category_brinsion_bhb.json').then(res => res.ok ? res.json() : Promise.resolve(null)),
                fetch('../output/B_category_brinsion_bf.json').then(res => res.ok ? res.json() : Promise.resolve(null)),
                fetch('../output/C_campisi_result_for_fof.json').then(res => res.ok ? res.json() : Promise.resolve(null)),
            ]);

            const allData = {
                equity: { BHB: processBrinsonData(equityBhb), BF: processBrinsonData(equityBf) },
                asset: { BHB: processBrinsonData(assetBhb), BF: processBrinsonData(assetBf) },
                campisi: processCampisiData(campisi)
            };
            
            // Initial Render
            renderEquityBrinson(allData.equity, 'BHB');
            renderAssetBrinson(allData.asset, 'BHB');
            renderCampisi(allData.campisi);

            // Setup Tab Listeners
            setupTabs('equity-brinson-tabs', (method) => renderEquityBrinson(allData.equity, method));
            setupTabs('asset-brinson-tabs', (method) => renderAssetBrinson(allData.asset, method));

        } catch (error) {
            console.error("Error loading or processing data:", error);
            document.getElementById('equity-brinson-content').innerHTML = `<p style="color:red;">权益类归因数据加载失败。请确保已成功运行 A01, A02, A03 脚本并生成了 output/A_..._fof_agg.json 文件。</p>`;
            document.getElementById('asset-brinson-content').innerHTML = `<p style="color:red;">大类归因数据加载失败。请确保已成功运行 B01, B02 脚本并生成了 output/B_...json 文件。</p>`;
            document.getElementById('campisi-content').innerHTML = `<p style="color:red;">债券归因数据加载失败。请确保已成功运行 C01, C02, C03 脚本并生成了 output/C_...json 文件。</p>`;
        }
    };

    // --- DATA PROCESSING ---
    const processBrinsonData = (longData) => {
        if (!longData || longData.length === 0) return null;
        const totals = {};
        const details = [];
        const totalKeys = ["Total_AR", "Total_SR", "Total_IR", "ER"];

        longData.forEach(item => {
            if (totalKeys.includes(item.index_code)) {
                totals[item.index_code] = item.index_value;
            } else {
                const match = item.index_code.match(/^(.*)_(AR|SR|IR)$/);
                if (match) {
                    const name = match[1];
                    const type = match[2];
                    let entry = details.find(d => d.name === name);
                    if (!entry) {
                        entry = { name: name, AR: 0, SR: 0, IR: 0 };
                        details.push(entry);
                    }
                    entry[type] = item.index_value;
                }
            }
        });
        return { totals, details };
    };
    
    const processCampisiData = (longData) => {
        if (!longData || longData.length === 0) return null;
        const data = {};
        longData.forEach(item => {
            data[item.index_code] = item.index_value;
        });
        
        // Ensure total_return is the sum of its components for consistency
        if (data.coupon_return !== undefined && data.capital_return !== undefined) {
            data.total_return = data.coupon_return + data.capital_return;
        }
        
        return data;
    };


    // --- RENDERING ---
    const formatPercent = (value) => {
        if (typeof value !== 'number') return 'N/A';
        return (value * 100).toFixed(1) + '%';
    }

    const createTreeNode = (label, value) => `
        <div class="tree-node">
            <div class="label">${label}</div>
            <div class="value">${formatPercent(value)}</div>
        </div>
    `;

    const renderBrinson = (containerId, data, method, titles) => {
        const container = document.getElementById(containerId);
        const modelData = data[method];
        if (!modelData) {
            container.innerHTML = `<p style="color:#dc2626;">无 ${method} 方法的归因数据。请先运行相关的Python计算脚本。</p>`;
            return;
        }

        const { totals, details } = modelData;
        const hasIR = method === 'BHB';

        const treeHtml = `
            <div class="tree-container">
                ${createTreeNode(titles.total, totals.ER || 0)}
                <div class="connector">
                    <div class="connector-line"></div>
                    <div class="connector-bracket"></div>
                    <div class="connector-line"></div>
                </div>
                <div class="tree-branch">
                    ${createTreeNode(titles.ar, totals.Total_AR || 0)}
                    ${createTreeNode(titles.sr, totals.Total_SR || 0)}
                    ${hasIR ? createTreeNode(titles.ir, totals.Total_IR || 0) : ''}
                </div>
            </div>
        `;

        const tableHtml = `
            <div class="detail-table-container">
                <table class="detail-table">
                    <thead>
                        <tr>
                            <th>${titles.detailName}</th>
                            <th>${titles.ar}</th>
                            <th>${titles.sr}</th>
                            ${hasIR ? `<th>${titles.ir}</th>` : ''}
                        </tr>
                    </thead>
                    <tbody>
                        ${details.map(d => `
                            <tr>
                                <td>${d.name}</td>
                                <td>${createBarCell(d.AR)}</td>
                                <td>${createBarCell(d.SR)}</td>
                                ${hasIR ? `<td>${createBarCell(d.IR)}</td>` : ''}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
        container.innerHTML = treeHtml + tableHtml;

        // --- Height Alignment Logic ---
        const treeContainer = container.querySelector('.tree-container');
        const tableContainer = container.querySelector('.detail-table-container');

        if (treeContainer && tableContainer) {
            const treeHeight = treeContainer.offsetHeight;
            tableContainer.style.height = `${treeHeight}px`;
        }
    };

    const renderEquityBrinson = (data, method) => renderBrinson('equity-brinson-content', data, method, {
        total: '超额收益',
        ar: '行业配置收益',
        sr: '个股选择收益',
        ir: '交互收益',
        detailName: '行业'
    });

    const renderAssetBrinson = (data, method) => renderBrinson('asset-brinson-content', data, method, {
        total: '超额收益',
        ar: '大类配置收益',
        sr: '个券选择收益',
        ir: '交互收益',
        detailName: '大类资产'
    });

    const renderCampisi = (data) => {
        const container = document.getElementById('campisi-content');
        if (!data) {
            container.innerHTML = `<p style="color:#dc2626;">无债券Campisi归因数据。请先运行相关的Python计算脚本。</p>`;
            return;
        }
        
        container.innerHTML = `
            <div class="tree-container">
                ${createTreeNode('债券收益', data.total_return || 0)}
                <div class="connector">
                    <div class="connector-line"></div>
                    <div class="connector-bracket-campisi"></div>
                    <div class="connector-line"></div>
                </div>
                <div class="tree-branch">
                    ${createTreeNode('票息收益', data.coupon_return || 0)}
                    <div class="tree-container">
                         ${createTreeNode('资本利得', data.capital_return || 0)}
                         <div class="connector">
                            <div class="connector-line"></div>
                            <div class="connector-bracket"></div>
                            <div class="connector-line"></div>
                        </div>
                        <div class="tree-branch">
                            ${createTreeNode('国债效应', data.duration_return || 0)}
                            ${createTreeNode('利差效应', data.spread_return || 0)}
                        </div>
                    </div>
                </div>
            </div>
        `;
    };

    const createBarCell = (value) => {
        if (typeof value !== 'number') return 'N/A';
        const maxBarWidth = 100; // in pixels
        const scale = 2000; // Adjust this factor to make bars larger or smaller
        const width = Math.min(Math.abs(value) * scale, maxBarWidth);
        const type = value >= 0 ? 'positive' : 'negative';
        
        const barHtml = `<div class="bar ${type}" style="width: ${width}px;"></div>`;
        const valueHtml = `<span class="value-text">${formatPercent(value)}</span>`;

        // For negative values, bar is on the left of the "zero axis"
        // A container with text-align right can simulate this
        if (type === 'negative') {
            return `<div class="bar-cell" style="justify-content: flex-end;">${valueHtml}${barHtml}</div>`;
        }
        return `<div class="bar-cell">${barHtml}${valueHtml}</div>`;
    };

    // --- UTILS ---
    const setupTabs = (tabContainerId, renderFn) => {
        const tabContainer = document.getElementById(tabContainerId);
        if (!tabContainer) return;
        tabContainer.addEventListener('click', (e) => {
            if (e.target.classList.contains('tab-button')) {
                const method = e.target.dataset.method;
                tabContainer.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
                e.target.classList.add('active');
                renderFn(method);
            }
        });
    };

    // --- INITIALIZATION ---
    fetchData();
});
