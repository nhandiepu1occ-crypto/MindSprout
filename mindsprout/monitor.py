"""
训练监控仪表盘

监控内容：
  ├── 子AI健康度（7项检测趋势）
  ├── AI父母行为（污染检测）
  ├── 记忆库状态（增长/命中率）
  ├── 系统资源（GPU/内存/磁盘）
  └── 告警（自动暂停+通知）

运行方式：python -m humanize_ai.monitor --port 8080
然后浏览器打开 http://localhost:8080
"""

from mindsprout.config import BASE

import json
import time
import re
import os
import sys
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler


# ============================================================
# 数据采集
# ============================================================

class TrainingMonitor:
    """训练过程的实时数据采集和告警"""
    
    def __init__(self):
        self.snapshots: List[Dict] = []
        self.alerts: List[Dict] = []
        self._paused = False
        self._pause_reason = ""
    
    def snapshot(self, data: Dict):
        """记录一次快照"""
        data["timestamp"] = datetime.now().isoformat()
        self.snapshots.append(data)
        
        # 自动告警检测
        self._check_alerts(data)
    
    def _check_alerts(self, data: Dict):
        """检查告警条件"""
        
        # 🚨 污染检测
        parent_feedback = data.get("parent_feedback", "")
        if self._detect_pollution(parent_feedback):
            self.alert(
                level="critical",
                source="AI父母",
                message="检测到AI父母可能在给答案而非只纠错！立即暂停检查。",
                detail=parent_feedback[:200],
            )
        
        # 🚨 综合分连续下降
        if len(self.snapshots) >= 3:
            recent = [s.get("composite_score", 0) for s in self.snapshots[-3:]]
            if recent[0] > recent[1] > recent[2]:
                self.alert(
                    level="critical",
                    source="子AI",
                    message=f"综合分连续3轮下降: {recent[0]:.2f}→{recent[1]:.2f}→{recent[2]:.2f}",
                )
        
        # 🟡 输出多样性下降
        diversity = data.get("output_diversity", 1.0)
        if diversity < 0.3:
            self.alert(
                level="warning",
                source="子AI",
                message=f"输出多样性过低({diversity:.2f})，可能过拟合到单一风格",
            )
        
        # 🔵 新类型不再产生
        new_types_rate = data.get("new_types_per_round", 0)
        if len(self.snapshots) >= 10 and new_types_rate == 0:
            recent_rate = sum(
                s.get("new_types_per_round", 0) for s in self.snapshots[-10:]
            ) / 10
            if recent_rate < 0.1:
                self.alert(
                    level="info",
                    source="类型表",
                    message="新类型产生速率接近0，经验源可能需要扩充",
                )
    
    def _detect_pollution(self, feedback: str) -> bool:
        """检测AI父母是否在给答案而非只纠错"""
        if not feedback:
            return False
        danger_patterns = [
            r'你应该\S{0,5}(写成|改成|说成)',
            r'(改成|换成|写成)这样',
            r'(比如|例如|像)：[^，。]{10,}',
            r'(参考|借鉴)这个',
            r'试试.{0,5}(写|说)：',
            r'应该这样(写|说|表达)',
        ]
        for pattern in danger_patterns:
            if re.search(pattern, feedback):
                return True
        return False
    
    def alert(self, level: str, source: str, message: str, detail: str = ""):
        """发出告警"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "source": source,
            "message": message,
            "detail": detail,
        }
        self.alerts.append(alert)
        
        emoji = {"critical": "🚨", "warning": "🟡", "info": "🔵"}
        print(f"\n{emoji.get(level, '❓')} [{level.upper()}] {source}: {message}")
        if detail:
            print(f"   详情: {detail}")
        
        # 严重告警 → 自动暂停
        if level == "critical":
            self._paused = True
            self._pause_reason = message
            print(f"\n⏸️ 训练已自动暂停。原因: {message}")
            print(f"   检查后运行 monitor.resume() 继续。")
    
    def resume(self):
        """手动恢复训练"""
        self._paused = False
        self._pause_reason = ""
        print("▶️ 训练已恢复")
    
    def status(self) -> Dict:
        """当前状态摘要"""
        if not self.snapshots:
            return {"status": "no_data"}
        
        latest = self.snapshots[-1]
        return {
            "status": "paused" if self._paused else "running",
            "pause_reason": self._pause_reason,
            "round": len(self.snapshots),
            "latest_composite": latest.get("composite_score", 0),
            "total_alerts": len(self.alerts),
            "critical_alerts": sum(1 for a in self.alerts if a["level"] == "critical"),
            "memory_nodes": latest.get("memory_nodes", 0),
            "type_table_size": latest.get("type_table_size", 0),
            "gpu_util": latest.get("gpu_utilization", 0),
        }
    
    def dashboard_data(self) -> Dict:
        """给Web仪表盘用的完整数据"""
        return {
            "status": self.status(),
            "alerts": self.alerts[-20:],
            "score_history": [
                {"round": i, "score": s.get("composite_score", 0)}
                for i, s in enumerate(self.snapshots[-50:])
            ],
            "memory_growth": [
                {"round": i, "nodes": s.get("memory_nodes", 0)}
                for i, s in enumerate(self.snapshots[-50:])
            ],
            "type_growth": [
                {"round": i, "types": s.get("type_table_size", 0)}
                for i, s in enumerate(self.snapshots[-50:])
            ],
            "parent_behavior": [
                {"round": i, "polluted": s.get("parent_pollution_detected", False)}
                for i, s in enumerate(self.snapshots[-50:])
            ],
        }


# ============================================================
# Web仪表盘
# ============================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="3">
<title>🎛️ Humanize-AI 训练监控</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: monospace; background: #0a0a0a; color: #0f0; padding: 20px; }
  h1 { color: #0ff; margin-bottom: 10px; }
  .status-bar { display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap; }
  .stat { background: #111; border: 1px solid #333; padding: 10px 15px; border-radius: 4px; }
  .stat .label { color: #888; font-size: 12px; }
  .stat .value { color: #0f0; font-size: 20px; font-weight: bold; }
  .alert { padding: 8px 12px; margin: 5px 0; border-radius: 3px; font-size: 13px; }
  .alert-critical { background: #300; border-left: 3px solid #f00; color: #f88; }
  .alert-warning { background: #330; border-left: 3px solid #ff0; color: #ff8; }
  .alert-info { background: #003; border-left: 3px solid #0ff; color: #8cf; }
  .chart { background: #111; border: 1px solid #333; padding: 10px; margin: 10px 0; border-radius: 4px; }
  .bar { height: 8px; background: #0f0; margin: 2px 0; border-radius: 2px; min-width: 1px; }
  .bar-danger { background: #f00; }
  .paused { color: #f00; animation: blink 1s infinite; }
  @keyframes blink { 50% { opacity: 0.3; } }
</style>
</head>
<body>
<h1>🎛️ Humanize-AI 训练监控</h1>
<div id="app">加载中...</div>
<script>
  async function refresh() {
    try {
      const res = await fetch('/data');
      const data = await res.json();
      render(data);
    } catch(e) {
      document.getElementById('app').innerHTML = '等待数据...';
    }
  }

  function render(data) {
    const s = data.status;
    const statusClass = s.status === 'paused' ? 'paused' : '';
    
    let html = `<div class="status-bar">
      <div class="stat"><div class="label">状态</div><div class="value ${statusClass}">${s.status === 'paused' ? '⏸️ 暂停' : '▶️ 运行中'}</div></div>
      <div class="stat"><div class="label">轮次</div><div class="value">${s.round || 0}</div></div>
      <div class="stat"><div class="label">综合分</div><div class="value">${(s.latest_composite || 0).toFixed(2)}</div></div>
      <div class="stat"><div class="label">告警</div><div class="value">${s.total_alerts || 0} (🚨${s.critical_alerts || 0})</div></div>
      <div class="stat"><div class="label">记忆节点</div><div class="value">${s.memory_nodes || 0}</div></div>
      <div class="stat"><div class="label">类型数</div><div class="value">${s.type_table_size || 0}</div></div>
      ${s.pause_reason ? `<div class="stat"><div class="label">暂停原因</div><div class="value" style="font-size:14px;color:#f00;">${s.pause_reason}</div></div>` : ''}
    </div>`;
    
    // 告警
    if (data.alerts && data.alerts.length > 0) {
      html += '<h3>🚨 告警</h3>';
      data.alerts.reverse().forEach(a => {
        html += `<div class="alert alert-${a.level}">[${a.level}] ${a.source}: ${a.message}</div>`;
      });
    }
    
    // 综合分趋势
    if (data.score_history && data.score_history.length > 0) {
      html += '<div class="chart"><h4>📊 综合分趋势</h4>';
      const maxH = 40;
      const maxScore = Math.max(...data.score_history.map(d => d.score), 0.1);
      data.score_history.forEach(d => {
        const w = Math.max(1, (d.score / maxScore) * 100);
        html += `<div class="bar" style="width:${w}%" title="轮${d.round}: ${d.score.toFixed(2)}"></div>`;
      });
      html += '</div>';
    }
    
    // 记忆增长
    if (data.memory_growth && data.memory_growth.length > 0) {
      html += '<div class="chart"><h4>🧠 记忆节点增长</h4>';
      const maxNodes = Math.max(...data.memory_growth.map(d => d.nodes), 1);
      data.memory_growth.forEach(d => {
        const w = Math.max(1, (d.nodes / maxNodes) * 100);
        html += `<div class="bar" style="width:${w}%" title="轮${d.round}: ${d.nodes}节点"></div>`;
      });
      html += '</div>';
    }
    
    // 父母行为
    if (data.parent_behavior && data.parent_behavior.length > 0) {
      html += '<div class="chart"><h4>👨‍🏫 AI父母行为</h4>';
      data.parent_behavior.forEach(d => {
        const cls = d.polluted ? 'bar-danger' : 'bar';
        html += `<div class="bar ${cls}" style="width:100%" title="轮${d.round}: ${d.polluted ? '⚠️污染' : '✅正常'}"></div>`;
      });
      html += '</div>';
    }
    
    document.getElementById('app').innerHTML = html;
  }
  
  refresh();
  setInterval(refresh, 3000);
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    monitor: 'TrainingMonitor' = None  # 由外部注入
    
    def do_GET(self):
        if self.path == '/':
            self._serve_html(DASHBOARD_HTML)
        elif self.path == '/data':
            self._serve_json(self.monitor.dashboard_data())
        elif self.path == '/status':
            self._serve_json(self.monitor.status())
        elif self.path == '/alerts':
            self._serve_json(self.monitor.alerts[-50:])
        else:
            self.send_response(404)
            self.end_headers()
    
    def _serve_html(self, html: str):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))
    
    def _serve_json(self, data: Dict):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))
    
    def log_message(self, format, *args):
        pass  # 静默HTTP日志


def start_dashboard(monitor: 'TrainingMonitor', port: int = 8080):
    """启动Web监控仪表盘"""
    DashboardHandler.monitor = monitor
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"\n📊 监控仪表盘: http://localhost:{port}")
    return server


# ============================================================
# CLI入口
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="训练监控仪表盘")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    
    monitor = TrainingMonitor()
    
    # 模拟一些数据
    import random
    for i in range(20):
        monitor.snapshot({
            "round": i,
            "composite_score": 0.5 + i * 0.02 + random.random() * 0.05,
            "output_diversity": max(0.2, 1.0 - i * 0.03),
            "new_types_per_round": max(0, 3 - i * 0.2),
            "memory_nodes": 100 + i * 50,
            "type_table_size": 5 + i,
            "parent_pollution_detected": i == 13,  # 第13轮检测到污染
            "parent_feedback": "你应该写成..." if i == 13 else "这句话太长了",
            "gpu_utilization": 60 + random.random() * 30,
        })
    
    start_dashboard(monitor, args.port)
    print("按 Ctrl+C 退出")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 监控已停止")
