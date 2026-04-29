from __future__ import annotations

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs


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
body{font-family:'Segoe UI','Microsoft YaHei',sans-serif;background:#0f1923;color:#d4dce8;min-height:100vh}
.header{background:#1a2736;border-bottom:2px solid #2a7de1;padding:12px 24px;display:flex;justify-content:space-between;align-items:center}
.header h1{font-size:20px;color:#5dade2}
.header .info{display:flex;gap:24px;font-size:13px;color:#8899aa}
.header .info span strong{color:#c8d6e5}
.status-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;animation:pulse 1.5s infinite}
.status-dot.connected{background:#2ecc71}
.status-dot.disconnected{background:#e74c3c}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.content{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:auto auto;gap:16px;padding:16px;max-height:calc(100vh - 60px)}
.chart-box{background:#1a2736;border-radius:8px;padding:12px;border:1px solid #243447}
.chart-box h3{font-size:14px;color:#7f9bb5;margin-bottom:8px}
.chart-box canvas{max-height:280px}
.log-panel{grid-column:1/-1;background:#1a2736;border-radius:8px;padding:12px;border:1px solid #243447;overflow:hidden;display:flex;flex-direction:column}
.log-panel h3{font-size:14px;color:#7f9bb5;margin-bottom:8px}
.log-container{flex:1;overflow-y:auto;font-family:'Cascadia Code','Fira Code','Consolas',monospace;font-size:12px;line-height:1.6;max-height:200px;background:#0d1520;border-radius:4px;padding:8px}
.log-line{padding:1px 0;border-bottom:1px solid #151f2a}
.log-line.episode{color:#5dade2}
.log-line.update{color:#f0c060}
.log-line.stage{color:#2ecc71;font-weight:bold}
.log-line.info{color:#8899aa}
.timestamp{color:#566b80;margin-right:8px}
@media(max-width:1100px){.content{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="header">
<h1>Vacuum Robot 训练监控</h1>
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
<div class="content">
<div class="chart-box"><h3>Policy Loss</h3><canvas id="chart-policy-loss"></canvas></div>
<div class="chart-box"><h3>Value Loss</h3><canvas id="chart-value-loss"></canvas></div>
<div class="chart-box"><h3>Entropy</h3><canvas id="chart-entropy"></canvas></div>
<div class="chart-box"><h3>Total Loss (Policy + Value)</h3><canvas id="chart-total-loss"></canvas></div>
<div class="chart-box"><h3>Episode Reward & EMA Cleaned</h3><canvas id="chart-reward"></canvas></div>
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
    policy_loss: '#e74c3c',
    value_loss: '#3498db',
    entropy: '#2ecc71',
    total_loss: '#e67e22',
    reward: '#f0c060',
    ema_cleaned: '#5dade2',
    cleaned: '#a569bd',
    steps: '#48c9b0',
    update_reward: '#f5b041'
};

const Y_PRECISION = {
    'Policy Loss': 4, 'Value Loss': 4, 'Entropy': 4, 'Total Loss': 4,
    'Episode Reward': 2, 'EMA Cleaned': 1,
    'Cleaned': 0, 'Steps': 0,
    'Mean Reward': 4
};

function createChart(canvasId, title) {
    const ctx = document.getElementById(canvasId).getContext('2d');
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
                x:{type:'linear', ticks:{color:'#566b80',font:{size:10}}, grid:{color:'#1e3040'}},
                y:{ticks:{color:'#566b80',font:{size:10}}, grid:{color:'#1e3040'}}
            },
            plugins: {
                legend:{labels:{color:'#7f9bb5',font:{size:11},usePointStyle:true,padding:16}},
                tooltip: {
                    enabled: true,
                    mode: 'nearest',
                    intersect: false,
                    backgroundColor: 'rgba(15,25,35,0.95)',
                    titleColor: '#8899aa',
                    bodyColor: '#d4dce8',
                    borderColor: '#2a7de1',
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

const rewardChart = createChart('chart-reward');
const rewardDs = addDataset(rewardChart, 'Episode Reward', COLORS.reward);
const emaDs = addDataset(rewardChart, 'EMA Cleaned', COLORS.ema_cleaned);

const episodeChart = createChart('chart-episode');
const cleanedDs = addDataset(episodeChart, 'Cleaned', COLORS.cleaned);
const stepsDs = addDataset(episodeChart, 'Steps', COLORS.steps);

const updateRewardChart = createChart('chart-update-reward');
const updateRewardDs = addDataset(updateRewardChart, 'Mean Reward', COLORS.update_reward);

let lossCounter = 0, rewardCounter = 0, episodeCounter = 0, updateRewardCounter = 0;

function addPoint(ds, counterRef, x, y) {
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
            } else if (evt.type === 'summary') {
                document.getElementById('h-step').textContent = d.step;
                if (d.fps) document.getElementById('h-fps').textContent = d.fps.toFixed(0);
                addLog('info', ts, 'Step=' + d.step + ' FPS=' + (d.fps||0).toFixed(0) + ' PolicyLoss=' + d.policy_loss.toFixed(4) + ' ValueLoss=' + d.value_loss.toFixed(4) + ' Entropy=' + d.entropy.toFixed(4) + ' ema_cleaned=' + d.ema_cleaned.toFixed(1));
            } else if (evt.type === 'stage') {
                document.getElementById('h-stage').textContent = d.stage_name;
                addLog('stage', ts, '>>> 进入阶段: ' + d.stage_name + ' at step ' + d.step);
            } else if (evt.type === 'info') {
                if (d.run_id) document.getElementById('h-run').textContent = d.run_id;
                if (d.seed != null) document.getElementById('h-seed').textContent = d.seed;
                addLog('info', ts, d.message || '');
            }
        }

        policyLossChart.update();
        valueLossChart.update();
        entropyChart.update();
        totalLossChart.update();
        rewardChart.update();
        episodeChart.update();
        updateRewardChart.update();

        document.getElementById('status-dot').className = 'status-dot connected';
        document.getElementById('status-text').textContent = '已连接';
    } catch (err) {
        document.getElementById('status-dot').className = 'status-dot disconnected';
        document.getElementById('status-text').textContent = '等待连接...';
        console.warn('Poll error:', err);
    }
}

window._lastEventTs = 0;
setInterval(poll, 1000);
poll();
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
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[Dashboard] Server started at http://{self.host}:{self.port}", flush=True)
        print(f"[Dashboard] 打开浏览器访问 http://localhost:{self.port}", flush=True)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        print("[Dashboard] Server stopped", flush=True)
