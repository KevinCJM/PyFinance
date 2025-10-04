export default function FullABC() {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-2">全量股票ABC择时分析</h1>
      <div className="text-sm text-gray-600 mb-4">说明：此页面用于批量运行A/B/C择时分析任务，支持与其他页面独立运行。后续可对接批处理API与任务进度展示。</div>
      <div className="bg-white rounded shadow p-4 text-sm text-gray-700">
        <p>功能占位：待接入后端批量分析接口。</p>
      </div>
    </div>
  );
}

