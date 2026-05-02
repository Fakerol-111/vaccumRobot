from __future__ import annotations

import json
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


class MetricsCollector:
    def __init__(self, max_events: int = 5000):
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._max_events = max_events
        self._run_info: dict[str, Any] = {}

    def set_run_info(self, info: dict[str, Any]) -> None:
        with self._lock:
            self._run_info = info

    def add_event(self, event_type: str, data: dict[str, Any]) -> None:
        with self._lock:
            self._events.append({
                "type": event_type,
                "timestamp": time.time(),
                "data": data,
            })
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

    def get_events_since(self, since_ts: float) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self._events if e["timestamp"] > since_ts]

    def get_all_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "run_info": self._run_info,
                "total_events": len(self._events),
            }


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>训练监控面板</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js">
</script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#0F172A;color:#E2E8F0;min-height:100vh}
.header{background:#1E293B;border-bottom:2px solid #38BDF8;padding:12px 24px;display:flex;justify-content:space-between;align-items:center}
.header h1{font-size:20px;color:#38BDF8}
.header .info{display:flex;gap:24px;font-size:13px;color:#94A3B8}
.header .info span strong{color:#E2E8F0}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;animation:pulse 1.5s infinite}
.status-dot.connected{background:#10B981}
.status-dot.disconnected{background:#F87171}
.algo-badge{display:inline-block;padding:2px 10px;border-radius:4px;font-weight:bold;font-size:12px;margin-left:8px}
.algo-badge.ppo{background:#F97316;color:#fff}
.algo-badge.grpo{background:#A78BFA;color:#fff}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.content{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px}
.chart-box{background:#1E293B;border-radius:8px;padding:12px;border:1px solid #334155}
.chart-box h3{font-size:14px;color:#94A3B8;margin-bottom:8px}
.chart-box canvas{max-height:280px}
.hidden{display:none!important}
.log-panel{grid-column:1/-1;background:#1E293B;border-radius:8px;padding:12px;border:1px solid #334155;overflow:hidden;display:flex;flex-direction:column}
.log-panel h3{font-size:14px;color:#94A3B8;margin-bottom:8px}
.log-container{flex:1;overflow-y:auto;font-family:'Cascadia Code','Fira Code','Consolas',monospace;font-size:12px;line-height:1.6;max-height:200px;background:#0F172A;border-radius:4px;padding:8px}
.log-line{padding:1px 0;border-bottom:1px solid #1E293B}
.log-line.episode{color:#38BDF8}
.log-line.update{color:#FACC15}
.log-line.stage{color:#10B981;font-weight:bold}
.log-line.info{color:#94A3B8}
.timestamp{color:#64748B;margin-right:8px}
@media(max-width:1100px){.content{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header">
<h1>Vacuum Robot 训练监控<span class="algo-badge ppo" id="algo-badge">PPO</span></h1>
<div class="info" id="header-info">
<span>Run: <strong id="h-run">-</strong></span>
<span>Seed: <strong id="h-seed">-</strong></span>
<span>步数: <strong id="h-step">0</strong></span>
<span>阶段: <strong id="h-stage">boot</strong></span>
<span>FPS: <strong id="h-fps">-</strong></span>
<span>Episodes: <strong id="h-eps">0</strong></span>
<span><span class="status-dot connected" id="status-dot"></span><span id="status-text">已连接</span></span>
</div>
</div>
<div class="content" id="chart-grid">
<div class="chart-box ppo-chart"><h3>Policy Loss</h3><canvas id="chart-policy-loss"></canvas></div>
<div class="chart-box ppo-chart"><h3>Value Loss</h3><canvas id="chart-value-loss"></canvas></div>
<div class="chart-box"><h3>Entropy</h3><canvas id="chart-entropy"></canvas></div>
<div class="chart-box ppo-chart"><h3>Total Loss (Policy + Value)</h3><canvas id="chart-total-loss"></canvas></div>
<div class="chart-box grpo-chart"><h3>Group Mean Score</h3><canvas id="chart-mean-score"></canvas></div>
<div class="chart-box grpo-chart"><h3>Group Std Score</h3><canvas id="chart-std-score"></canvas></div>
<div class="chart-box grpo-chart"><h3>KL Divergence</h3><canvas id="chart-kl"></canvas></div>
<div class="chart-box grpo-chart"><h3>Grad Norm</h3><canvas id="chart-grad-norm"></canvas></div>
<div class="chart-box grpo-chart"><h3>Total Loss</h3><canvas id="chart-total-loss-grpo"></canvas></div>
<div class="chart-box"><h3>Episode Reward &amp; EMA Cleaned</h3><canvas id="chart-reward"></canvas></div>
<div class="chart-box"><h3>Episode 指标 (Cleaned / Steps)</h3><canvas id="chart-episode"></canvas></div>
<div class="chart-box"><h3>Update Mean Reward</h3><canvas id="chart-update-reward"></canvas></div>
<div class="chart-box"></div>
<div class="log-panel">
<h3>实时日志</h3>
<div class="log-container" id="log-container"></div>
</div>
</div>
<script>
const MAX_POINTS = 300;
const COLORS = {
    policy_loss: '#F97316',
    value_loss: '#38BDF8',
    entropy: '#10B981',
    total_loss: '#F97316',
    reward: '#FACC15',
    ema_cleaned: '#38BDF8',
    cleaned: '#A78BFA',
    steps: '#10B981',
    update_reward: '#FACC15',
    mean_score: '#A78BFA',
    std_score: '#38BDF8',
    kl: '#FB7185',
    grpo_total_loss: '#F97316',
    policy_loss_grpo: '#FB7185',
    grad_norm: '#34D399'
};

const Y_PRECISION = {
    'Policy Loss': 4, 'Value Loss': 4, 'Entropy': 4, 'Total Loss': 4,
    'Episode Reward': 2, 'EMA Cleaned': 1,
    'Cleaned': 0, 'Steps': 0,
    'Mean Reward': 4,
    'Mean Score': 4, 'Std Score': 4, 'KL Divergence': 6,
    'Grad Norm': 6
};

let currentAlgo = 'ppo';

function switchAlgo(name) {
    currentAlgo = name;
    document.querySelectorAll('.ppo-chart').forEach(el => el.classList.toggle('hidden', name !== 'ppo'));
    document.querySelectorAll('.grpo-chart').forEach(el => el.classList.toggle('hidden', name !== 'grpo'));
    const badge = document.getElementById('algo-badge');
    badge.textContent = name.toUpperCase();
    badge.className = 'algo-badge ' + name;
    setTimeout(function() {
        document.querySelectorAll('.chart-box:not(.hidden) canvas').forEach(function(c) {
            var ch = Chart.getChart(c);
            if (ch) ch.resize();
        });
    }, 100);
}

function createChart(canvasId, title) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    const ctx = canvas.getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {datasets: []},
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {duration: 100},
            interaction: {
                mode: 'nearest',
                intersect: false,
                axis: 'x'
            },
            scales: {
                x:{type:'linear', ticks:{color:'#64748B',font:{size:10}}, grid:{color:'#1E293B'}},
                y:{ticks:{color:'#64748B',font:{size:10}}, grid:{color:'#1E293B'}}
            },
            plugins: {
                legend:{labels:{color:'#94A3B8',font:{size:11},usePointStyle:true,padding:16}},
                tooltip: {
                    enabled: true,
                    mode: 'nearest',
                    intersect: false,
                    backgroundColor: 'rgba(15,23,42,0.95)',
                    titleColor: '#94A3B8',
                    bodyColor: '#E2E8F0',
                    borderColor: '#38BDF8',
                    borderWidth: 1,
                    padding: 8,
                    titleFont: {size: 11},
                    bodyFont: {size: 12},
                    callbacks: {
                        title: function(items) {
                            if (!items.length) return '';
                            return title + '  #' + items[0].parsed.x;
                        },
                        label: function(ctx) {
                            return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(Y_PRECISION[ctx.dataset.label] || 4);
                        }
                    }
                }
            }
        }
    });
}

function addDataset(chart, label, color) {
    if (!chart) return null;
    const ds = {
        label, borderColor: color, backgroundColor: color + '20',
        data: [], pointRadius: 0, pointHitRadius: 10, pointHoverRadius: 5,
        pointHoverBackgroundColor: color, pointHoverBorderColor: '#fff',
        pointHoverBorderWidth: 1, borderWidth: 1.5, tension: 0.1,
        parsing: false
    };
    chart.data.datasets.push(ds);
    return ds;
}

const policyLossChart = createChart('chart-policy-loss');
const policyLossDs = addDataset(policyLossChart, 'Policy Loss', COLORS.policy_loss);

const valueLossChart = createChart('chart-value-loss');
const valueLossDs = addDataset(valueLossChart, 'Value Loss', COLORS.value_loss);

const entropyChart = createChart('chart-entropy');
const entropyDs = addDataset(entropyChart, 'Entropy', COLORS.entropy);

const totalLossChart = createChart('chart-total-loss');
const totalLossDs = addDataset(totalLossChart, 'Total Loss', COLORS.total_loss);

const meanScoreChart = createChart('chart-mean-score');
const meanScoreDs = addDataset(meanScoreChart, 'Mean Score', COLORS.mean_score);

const stdScoreChart = createChart('chart-std-score');
const stdScoreDs = addDataset(stdScoreChart, 'Std Score', COLORS.std_score);

const klChart = createChart('chart-kl');
const klDs = addDataset(klChart, 'KL Divergence', COLORS.kl);

const gradNormChart = createChart('chart-grad-norm');
const gradNormDs = addDataset(gradNormChart, 'Grad Norm', COLORS.grad_norm);

const grpoTotalLossChart = createChart('chart-total-loss-grpo');
const grpoTotalLossDs = addDataset(grpoTotalLossChart, 'Total Loss', COLORS.grpo_total_loss);
const grpoPolicyLossDs = addDataset(grpoTotalLossChart, 'Policy Loss', COLORS.policy_loss_grpo);

const rewardChart = createChart('chart-reward');
const rewardDs = addDataset(rewardChart, 'Episode Reward', COLORS.reward);
const emaDs = addDataset(rewardChart, 'EMA Cleaned', COLORS.ema_cleaned);

const episodeChart = createChart('chart-episode');
const cleanedDs = addDataset(episodeChart, 'Cleaned', COLORS.cleaned);
const stepsDs = addDataset(episodeChart, 'Steps', COLORS.steps);

const updateRewardChart = createChart('chart-update-reward');
const updateRewardDs = addDataset(updateRewardChart, 'Mean Reward', COLORS.update_reward);

let lossCounter = 0, episodeCounter = 0, updateRewardCounter = 0, grpoCounter = 0;

function addPoint(ds, counterRef, x, y) {
    if (!ds) return;
    ds.data.push({x, y});
    if (ds.data.length > MAX_POINTS) ds.data.shift();
}

function addLog(className, timestamp, msg) {
    const container = document.getElementById('log-container');
    const div = document.createElement('div');
    div.className = 'log-line ' + className;
    div.innerHTML = '<span class="timestamp">' + timestamp + '</span>' + msg;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    if (container.children.length > 200) container.firstChild.remove();
}

function formatTime(ts) {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString('zh-CN');
}

async function initAlgo() {
    try {
        const resp = await fetch('/api/info');
        if (!resp.ok) return;
        const info = await resp.json();
        if (info && info.algo) switchAlgo(info.algo);
    } catch (e) {}
}

async function poll() {
    try {
        const since = window._lastEventTs || 0;
        const resp = await fetch('/api/data?since=' + since);
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const events = await resp.json();
        if (!events || events.length === 0) return;

        window._lastEventTs = events[events.length - 1].timestamp;

        for (const evt of events) {
            const ts = formatTime(evt.timestamp);
            const d = evt.data;

            if (evt.type === 'episode') {
                episodeCounter++;
                addPoint(rewardDs, null, episodeCounter, d.reward);
                addPoint(emaDs, null, episodeCounter, d.ema_cleaned);
                addPoint(cleanedDs, null, episodeCounter, d.cleaned);
                addPoint(stepsDs, null, episodeCounter, d.steps);
                document.getElementById('h-eps').textContent = d.episode;
                const mapTag = d.map_name ? '[' + d.map_name + '] ' : '';
                addLog('episode', ts, mapTag + 'cleaned=' + d.cleaned + ' steps=' + d.steps + ' charges=' + d.charges + ' reward=' + d.reward.toFixed(1) + ' ema=' + d.ema_cleaned.toFixed(1));
            } else if (evt.type === 'update') {
                lossCounter++;
                const idx = d.update_idx;
                addPoint(policyLossDs, null, idx, d.policy_loss);
                addPoint(valueLossDs, null, idx, d.value_loss);
                addPoint(entropyDs, null, idx, d.entropy);
                addPoint(totalLossDs, null, idx, d.policy_loss + d.value_loss);
                addPoint(updateRewardDs, null, idx, d.reward);
                addLog('update', ts, 'update=' + idx + ' policy_loss=' + d.policy_loss.toFixed(4) + ' value_loss=' + d.value_loss.toFixed(4) + ' entropy=' + d.entropy.toFixed(4) + ' total_loss=' + (d.policy_loss + d.value_loss).toFixed(4) + ' reward=' + d.reward.toFixed(4));
            } else if (evt.type === 'group_update') {
                grpoCounter++;
                const idx = grpoCounter;
                addPoint(meanScoreDs, null, idx, d.mean_score);
                addPoint(stdScoreDs, null, idx, d.std_score);
                addPoint(klDs, null, idx, d.kl);
                addPoint(gradNormDs, null, idx, d.grad_norm);
                addPoint(entropyDs, null, idx, d.entropy);
                addPoint(grpoTotalLossDs, null, idx, d.total_loss);
                addPoint(grpoPolicyLossDs, null, idx, d.policy_loss);
                addPoint(updateRewardDs, null, idx, d.mean_score);
                addLog('update', ts, 'group_update=' + idx + ' total_loss=' + d.total_loss.toFixed(4) + ' policy_loss=' + d.policy_loss.toFixed(4) + ' mean_score=' + d.mean_score.toFixed(4) + ' std_score=' + d.std_score.toFixed(4) + ' kl=' + d.kl.toFixed(6) + ' grad_norm=' + (d.grad_norm||0).toFixed(6) + ' entropy=' + (d.entropy||0).toFixed(4));
            } else if (evt.type === 'summary') {
                document.getElementById('h-step').textContent = d.step;
                if (d.fps) document.getElementById('h-fps').textContent = d.fps.toFixed(0);
                addLog('info', ts, 'Step=' + d.step + ' FPS=' + (d.fps||0).toFixed(0) + ' ema_cleaned=' + d.ema_cleaned.toFixed(1) + ' episodes=' + d.episodes);
            } else if (evt.type === 'stage') {
                document.getElementById('h-stage').textContent = d.stage_name;
                addLog('stage', ts, '>>> 进入阶段: ' + d.stage_name + ' at step ' + d.step);
            } else if (evt.type === 'info') {
                if (d.run_id) document.getElementById('h-run').textContent = d.run_id;
                if (d.seed != null) document.getElementById('h-seed').textContent = d.seed;
                if (d.algo) switchAlgo(d.algo);
                addLog('info', ts, d.message || '');
            }
        }

        [policyLossChart, valueLossChart, entropyChart, totalLossChart,
         meanScoreChart, stdScoreChart, klChart, gradNormChart, grpoTotalLossChart,
         rewardChart, episodeChart, updateRewardChart].forEach(function(ch) { if (ch) ch.update(); });

        document.getElementById('status-dot').className = 'status-dot connected';
        document.getElementById('status-text').textContent = '已连接';
    } catch (err) {
        document.getElementById('status-dot').className = 'status-dot disconnected';
        document.getElementById('status-text').textContent = '等待连接...';
        console.warn('Poll error:', err);
    }
}

document.querySelectorAll('.grpo-chart').forEach(function(el) { el.classList.add('hidden'); });

window._lastEventTs = 0;
initAlgo().then(poll);
setInterval(poll, 1000);
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    collector: MetricsCollector = None

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._serve_html()
        elif path == "/api/data":
            self._serve_api_data(params)
        elif path == "/api/info":
            self._serve_api_info()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def _serve_html(self):
        html = DASHBOARD_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def _serve_api_data(self, params: dict):
        since = 0.0
        if "since" in params:
            try:
                since = float(params["since"][0])
            except (ValueError, IndexError):
                pass

        events = self.collector.get_events_since(since)
        payload = json.dumps(events, ensure_ascii=False).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def _serve_api_info(self):
        summary = self.collector.get_summary()
        run_info = summary.get("run_info", {})
        payload = json.dumps(run_info, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)


class DashboardServer:
    def __init__(self, collector: MetricsCollector, host: str = "0.0.0.0", port: int = 8088):
        self.collector = collector
        self.host = host
        self.port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        DashboardHandler.collector = self.collector
        self._server = HTTPServer((self.host, self.port), DashboardHandler)

        def _serve() -> None:
            try:
                self._server.serve_forever()
            except OSError:
                pass  # 关闭时 select 可能抛 OSError，忽略即可

        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()
        logger.info("Dashboard Server started at http://%s:%s", self.host, self.port)
        logger.info("打开浏览器访问 http://localhost:%s", self.port)

    def stop(self) -> None:
        # 先 shutdown → serve_forever 收到信号退出 select 循环
        if self._server is not None:
            try:
                self._server.shutdown()
            except OSError:
                pass  # 可能已经关闭
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._server is not None:
            self._server.server_close()
            self._server = None
        logger.info("Dashboard Server stopped")


def create_dashboard(dashboard_config: dict) -> tuple[MetricsCollector | None, DashboardServer | None]:
    collector = None
    dashboard_server = None
    if dashboard_config["enabled"]:
        collector = MetricsCollector()
        dashboard_server = DashboardServer(
            collector,
            host=dashboard_config["host"],
            port=dashboard_config["port"],
        )
        dashboard_server.start()
    return collector, dashboard_server


__all__ = ["MetricsCollector", "DashboardServer", "DashboardHandler", "DASHBOARD_HTML", "create_dashboard"]
