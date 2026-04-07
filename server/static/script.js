// Dashboard Script v1.0.1 - drone-env
// ═══════════════════════════════════════════════════════
//  CONFIG
// ═══════════════════════════════════════════════════════
const BASE = ''; 
const DIRECTIONS = ['UP','DOWN','LEFT','RIGHT','WAIT'];
const EMOJI = {
    drone: "🚁",
    road: "🛣️",
    building: "🏢",
    tree: "🌳",
    obstacle: "🚧",
    delivery: "📦",
    done_del: "✅"
};

// ═══════════════════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════════════════
let currentTask = 'easy_delivery';
let autoTimer   = null;
let logTimer    = null;
let obs         = null;
let rewardHistory = [];
let stepHistory   = []; // For CSV export
let lastLogs      = "";
let lastTerminalLogs = "";
let autoActive = false;
let startTime = null;

// ═══════════════════════════════════════════════════════
//  REWARD CHART
// ═══════════════════════════════════════════════════════
let rewardChart = null;
function initChart() {
    try {
        const ctx = document.getElementById('rewardChart').getContext('2d');
        rewardChart = new Chart(ctx, {
          type: 'line',
          data: {
            labels: [],
            datasets: [{
              label: 'Step Reward',
              data: [],
              borderColor: 'rgba(0,229,255,0.8)',
              backgroundColor: 'rgba(0,229,255,0.07)',
              borderWidth: 2,
              fill: true,
              tension: 0.4,
              pointRadius: 0,
            }]
          },
          options: {
            animation: false,
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { display: false },
              y: {
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: 'rgba(255,255,255,0.3)', font: { size: 9 } },
                grace: '5%'  // Add some breathing room for small changes
              }
            },
            plugins: { legend: { display: false } }
          }
        });
    } catch(e) {
        console.warn("Chart.js failed to load. Chart will be disabled.", e);
        rewardChart = null;
    }
}

// ═══════════════════════════════════════════════════════
//  UI UTILS
// ═══════════════════════════════════════════════════════
function showLoading() { document.getElementById('processing').style.display = 'flex'; }
function hideLoading() { document.getElementById('processing').style.display = 'none'; }

function updateUI(data) {
    obs = data;
    renderGrid(obs);

    // Telemetry updates
    document.getElementById('statStep').textContent = obs.step_count;
    document.getElementById('statReward').textContent = obs.reward_last.toFixed(3);
    document.getElementById('statDel').textContent = `${obs.deliveries_done}/${obs.deliveries_total}`;
    
    // Performance Score
    const scoreEl = document.getElementById('statScore');
    if (scoreEl) scoreEl.textContent = obs.score.toFixed(3);
    
    
    // Delivery Progress Bar
    const delBarFill = document.getElementById('delBarFill');
    const delBarlabel = document.getElementById('delBarlabel');
    if (delBarFill && delBarlabel) {
        const progress = obs.deliveries_total > 0 ? (obs.deliveries_done / obs.deliveries_total) * 100 : 0;
        delBarFill.style.width = progress + '%';
        delBarlabel.textContent = `${obs.deliveries_done} / ${obs.deliveries_total}`;
        
        // Battery Critical Effect
        const batEl = document.getElementById('batPct');
        if (batEl) {
            if (obs.battery < 0.2) {
                batEl.classList.add('critical-flash');
            } else {
                batEl.classList.remove('critical-flash');
            }
        }
    }
    
    // Battery
    const batPct = Math.max(0, Math.round(obs.battery * 100));
    document.getElementById('batPct').textContent = batPct + '%';
    const batFill = document.getElementById('batFill');
    batFill.style.width = batPct + '%';
    batFill.className = 'battery-fill' + (batPct < 25 ? ' low' : '');

    // Chart & History
    rewardHistory.push(obs.reward_last);
    if(rewardHistory.length > 50) rewardHistory.shift();
    if(rewardChart) {
        rewardChart.data.labels = rewardHistory.map((_, i) => i);
        rewardChart.data.datasets[0].data = rewardHistory;
        rewardChart.update('none');
    }

    // Store for CSV
    stepHistory.push({
        step: obs.step_count,
        x: obs.drone_x,
        y: obs.drone_y,
        reward: obs.reward_last.toFixed(4),
        total_reward: obs.reward_total.toFixed(4),
        score: obs.score.toFixed(4),
        battery: batPct,
        message: obs.message
    });

    // Message bar
    const msgEl = document.getElementById('msgBar');
    msgEl.textContent = obs.message;
    msgEl.className = 'msg-bar' + (obs.done ? (obs.deliveries_done === obs.deliveries_total ? ' good' : ' bad') : '');

    // Log
    addLog(obs);

    // Auto-stop on battery zero or done
    if (obs.done || obs.battery <= 0) {
        stopAuto();
        showCompletionPopup(obs);
    }
    
    // Live JSON Telemetry Stream
    updateLiveTelemetry(obs);
}

function updateLiveTelemetry(obs) {
    const consoleEl = document.getElementById('memoryConsole');
    if (!consoleEl) return;
    
    // Create a clean telemetry slice
    const telemetry = {
        step: obs.step_count,
        pos: `(${obs.drone_x}, ${obs.drone_y})`,
        reward: parseFloat(obs.reward_last.toFixed(4)),
        total_reward: parseFloat(obs.reward_total.toFixed(4)),
        battery: `${Math.round(obs.battery * 100)}%`,
        status: obs.message
    };
    
    // Append JSON line
    const line = document.createElement('div');
    line.className = 'console-line';
    line.style.color = '#00e5ff'; // Cyan for JSON
    line.innerHTML = `> ${JSON.stringify(telemetry)}`;
    
    // If it's the first real log, clear the "Awaiting" message
    if (consoleEl.children.length === 1 && consoleEl.innerHTML.includes("Awaiting")) {
        consoleEl.innerHTML = "";
    }
    
    consoleEl.appendChild(line);
    
    // Keep last 50 lines to prevent lag
    if (consoleEl.children.length > 50) {
        consoleEl.firstChild.remove();
    }
    
    consoleEl.scrollTop = consoleEl.scrollHeight;
}

function addLog(obs) {
    const list = document.getElementById('logList');
    if (!list) return;
    const item = document.createElement('div');
    const r = obs.reward_last;
    item.className = 'log-item' + (r > 0 ? ' good' : r < -0.1 ? ' bad' : '');
    item.innerHTML = `<span class="log-step">#${obs.step_count}</span> <span style="flex:1">${obs.message}</span> <span>${r.toFixed(3)}</span>`;
    list.prepend(item);
    if(list.children.length > 20) list.lastChild.remove();
}

function renderGrid(obs) {
    const wrap = document.getElementById('gridWrap');
    if (!wrap) return;
    const grid = obs.grid;
    const W = obs.grid_width;
    const H = obs.grid_height;

    // Adjust cell size dynamically based on container width
    const parentWidth = wrap.parentElement.clientWidth;
    const paddingBuffer = window.innerWidth < 480 ? 20 : 48; 
    const maxW = parentWidth - paddingBuffer; 
    const cellPx = Math.max(16, Math.min(36, Math.floor(maxW / W)));
    document.documentElement.style.setProperty('--cell', cellPx + 'px');

    let html = '';
    for (let y = 0; y < grid.length; y++) {
        html += '<div class="grid-row">';
        const cells = splitEmojis(grid[y]);
        for (let x = 0; x < cells.length; x++) {
            const ch = cells[x];
            html += `<div class="grid-cell" data-x="${x}" data-y="${y}">${ch}</div>`;
        }
        html += '</div>';
    }
    wrap.innerHTML = html;
    document.getElementById('gridInfo').textContent = `${W}×${H}`;
}

function splitEmojis(str) {
  if (typeof Intl !== 'undefined' && Intl.Segmenter) {
    const seg = new Intl.Segmenter();
    return [...seg.segment(str)].map(s => s.segment);
  }
  return [...str];
}

// ═══════════════════════════════════════════════════════
//  API ACTIONS
// ═══════════════════════════════════════════════════════
async function doReset() {
    showLoading();
    stopAuto();
    try {
        const r = await fetch(`${BASE}/reset`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ task_name: currentTask })
        });
        const data = await r.json();
        rewardHistory = [];
        stepHistory = [];
        startTime = Date.now();
        const logList = document.getElementById('logList');
        if(logList) logList.innerHTML = '';
        updateUI(data);
    } finally {
        hideLoading();
    }
}

async function doStep(dir) {
    if (obs?.done) return;
    try {
        const r = await fetch(`${BASE}/step`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ direction: dir })
        });
        const data = await r.json();
        updateUI(data);
        flashBtn(dir);
    } catch(e) { console.error(e); }
}

function flashBtn(dir) {
    const map = { UP:'btnUp', DOWN:'btnDown', LEFT:'btnLeft', RIGHT:'btnRight', WAIT:'btnWait' };
    const id = map[dir];
    if (!id) return;
    const btn = document.getElementById(id);
    if (!btn) return;
    btn.style.borderColor = 'var(--blue)';
    btn.style.boxShadow = '0 0 15px var(--blue)';
    btn.style.transition = 'all 0.1s';
    setTimeout(() => {
        btn.style.borderColor = '';
        btn.style.boxShadow = '';
    }, 200);
}

async function doAnalyse() {
    showLoading();
    try {
        const r = await fetch(`${BASE}/analyse/${currentTask}`);
        const data = await r.json();
        renderAnalytics(data);
    } finally {
        hideLoading();
    }
}

function renderAnalytics(data) {
    const grid = document.getElementById('analyticsGrid');
    if (data.error || data.message === 'Click Analyse') {
        grid.innerHTML = `<div class="analytics-row"><span class="analytics-key">INFO</span><span class="analytics-val" style="color:var(--dim)">${data.error || data.message || 'No data'}</span></div>`;
        return;
    }
    const rows = [
        ['Episodes',      data.total_episodes,                 'cyan'],
        ['Avg Steps',     data.avg_steps,                      ''],
        ['Avg Deliveries',data.avg_deliveries,                 'green'],
        ['Avg Reward',    data.avg_reward?.toFixed(3),         (data.avg_reward || 0) > 0 ? 'green' : 'red'],
    ];
    grid.innerHTML = rows.map(([k,v,c]) =>
        `<div class="analytics-row">
            <span class="analytics-key">${k}</span>
            <span class="analytics-val ${c}">${v ?? '—'}</span>
        </div>`
    ).join('');

    // Action distribution bars
    const dist = data.action_distribution || {};
    const total = Object.values(dist).reduce((a,b) => a+b, 0) || 1;
    ['UP','DOWN','LEFT','RIGHT','WAIT'].forEach(a => {
        const bar = document.getElementById('ab-' + a);
        if (bar) {
            const h = Math.round((dist[a] || 0) / total * 36);
            bar.style.height = h + 'px';
        }
    });
}

// ═══════════════════════════════════════════════════════
//  CSV DOWNLOAD
// ═══════════════════════════════════════════════════════
function downloadCSV() {
    if (stepHistory.length === 0) {
        alert("No telemetry data to download yet!");
        return;
    }
    // Sort by step to ensure chronological order
    const sortedHistory = [...stepHistory].sort((a,b) => a.step - b.step);
    
    const headers = Object.keys(sortedHistory[0]).join(',');
    const rows = sortedHistory.map(row => Object.values(row).join(','));
    const csvContent = "data:text/csv;charset=utf-8," + headers + "\n" + rows.join("\n");
    
    const link = document.createElement("a");
    link.setAttribute("href", encodeURI(csvContent));
    link.setAttribute("download", `drone_mission_${currentTask}_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function downloadGraph() {
    if (!rewardChart || rewardHistory.length === 0) {
        alert("Mission telemetry graph not available! Please start the engine to generate data.");
        return;
    }
    try {
        const canvas = document.getElementById('rewardChart');
        const url = canvas.toDataURL("image/png");
        const link = document.createElement('a');
        link.download = `drone_rewards_${currentTask}_${Date.now()}.png`;
        link.href = url;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    } catch (e) {
        console.error("Graph download failed:", e);
        alert("Failed to download graph. See console for details.");
    }
}

// ═══════════════════════════════════════════════════════
//  AUTO-PILOT
// ═══════════════════════════════════════════════════════
function toggleAuto() {
    autoActive = document.getElementById('autoToggle').checked;
    if (autoActive) autoLoop();
}

async function autoLoop() {
    if (!autoActive || obs?.done) return;
    try {
        const r = await fetch(`${BASE}/predict`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(obs)
        });
        const { direction } = await r.json();
        if (autoActive) {
            await doStep(direction || 'WAIT');
            // If the model says WAIT, add a slightly longer pause to avoid spamming
            const delay = (direction === 'WAIT' || !direction) ? 1200 : 600;
            setTimeout(autoLoop, delay);
        }
    } catch (e) {
        if (autoActive) setTimeout(autoLoop, 2000);
    }
}

function startAuto() {
    autoActive = true;
    const toggle = document.getElementById('autoToggle');
    if(toggle) toggle.checked = true;
    
    document.querySelector('.status-dot').classList.add('active');
    document.querySelector('.status-indicator span').textContent = 'NEURAL_ENGINE: ACTIVE';
    autoLoop();
}

function stopAuto() {
    autoActive = false;
    const toggle = document.getElementById('autoToggle');
    if(toggle) toggle.checked = false;
    
    const dot = document.querySelector('.status-dot');
    if(dot) dot.classList.remove('active');
    const lbl = document.querySelector('.status-indicator span');
    if(lbl) lbl.textContent = 'NEURAL_ENGINE: READY';
}

// ═══════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════
window.onload = async () => {
    initChart();
    
    // Task selector
    document.getElementById('taskGroup').onclick = (e) => {
        const btn = e.target.closest('.task-btn');
        if (btn) {
            document.querySelectorAll('.task-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentTask = btn.dataset.task;
            doReset();
        }
    };

    // Keyboard
    document.addEventListener('keydown', e => {
        const map = { ArrowUp:'UP', ArrowDown:'DOWN', ArrowLeft:'LEFT', ArrowRight:'RIGHT' };
        if (map[e.code]) {
            e.preventDefault();
            doStep(map[e.code]);
        }
    });

    await doReset();
    startLogPolling();
    startTerminalLogPolling();
    
    const loading = document.getElementById('loading');
    if (loading) {
        loading.style.opacity = '0';
        setTimeout(() => loading.style.display = 'none', 400);
    }

    // Dynamic resize listener to keep grid scaling interactive
    window.addEventListener('resize', () => {
        if (obs) renderGrid(obs);
    });
};

// ═══════════════════════════════════════════════════════
//  NEURAL LOG POLLING
// ═══════════════════════════════════════════════════════
function startLogPolling() {
    if(logTimer) clearInterval(logTimer);
    logTimer = setInterval(async () => {
        try {
            const r = await fetch(`${BASE}/logs`);
            const { logs } = await r.json();
            const consoleEl = document.getElementById('neuralConsole');
            if(!consoleEl || !logs || logs.length === 0) return;

            const newContent = logs.join("");
            if(newContent === lastLogs) return;
            lastLogs = newContent;

            consoleEl.innerHTML = logs.map(line => {
                let cls = "";
                if(line.includes("Reward")) cls = "success";
                if(line.includes("complete")) cls = "info";
                if(line.includes("saved")) cls = "info";
                return `<div class="console-line ${cls}">> ${line}</div>`;
            }).join("");
            
            consoleEl.scrollTop = consoleEl.scrollHeight;
        } catch(e) {}
    }, 2000);
}

// ═══════════════════════════════════════════════════════
//  ANTIGRAVITY TERMINAL POLLING
// ═══════════════════════════════════════════════════════
function startTerminalLogPolling() {
    setInterval(async () => {
        try {
            const r = await fetch(`${BASE}/terminal_logs`);
            const { logs } = await r.json();
            const consoleEl = document.getElementById('terminalConsole');
            if(!consoleEl || !logs || logs.length === 0) return;

            const newContent = logs.join("\n");
            if(newContent === lastTerminalLogs && consoleEl.innerHTML !== "") return;
            lastTerminalLogs = newContent;

            consoleEl.innerHTML = logs.map(line => {
                let cls = "";
                if(line.includes("POST")) cls = "success";
                if(line.includes("GET")) cls = "info";
                if(line.includes(" 404 ") || line.includes(" 500 ")) cls = "warn";
                return `<div class="console-line ${cls}">> ${line}</div>`;
            }).join("");
            
            consoleEl.scrollTop = consoleEl.scrollHeight;
        } catch(e) {}
    }, 1000); // Poll slightly faster for real-time feel
}
// ═══════════════════════════════════════════════════════
//  COMPLETION MODAL
// ═══════════════════════════════════════════════════════
function showCompletionPopup(obs) {
    const modal = document.getElementById('completionModal');
    if (!modal) return;

    const isSuccess = obs.deliveries_done === obs.deliveries_total;
    document.getElementById('summaryStatus').textContent = isSuccess ? "MISSION LOG: SUCCESS" : "MISSION LOG: FAILED";
    document.getElementById('summaryStatus').style.color = isSuccess ? "var(--green)" : "var(--red)";

    document.getElementById('summaryScore').textContent = obs.score.toFixed(3);
    document.getElementById('summaryDel').textContent = `${obs.deliveries_done}/${obs.deliveries_total}`;
    document.getElementById('summarySteps').textContent = obs.step_count;
    document.getElementById('summaryReward').textContent = obs.reward_total.toFixed(3);
    
    const avg = obs.step_count > 0 ? (obs.reward_total / obs.step_count).toFixed(4) : "0.000";
    document.getElementById('summaryAvg').textContent = avg;

    const delRatio = obs.deliveries_total > 0 ? (obs.deliveries_done / obs.deliveries_total) : 0;
    const stepRatio = obs.max_steps > 0 ? (1 - obs.step_count / obs.max_steps) : 0;
    const batRatio = obs.battery; // Already 0.0-1.0
    
    // Dynamic Efficiency: 75% Completion, 15% Battery, 10% Speed
    const efficiency = (delRatio * 75) + (batRatio * 15) + (stepRatio * 10);
    document.getElementById('summaryEfficiency').textContent = efficiency.toFixed(1) + "%";

    const elapsed = startTime ? ((Date.now() - startTime) / 1000).toFixed(1) : "0.0";
    document.getElementById('summaryTime').textContent = elapsed + "s";

    const list = document.getElementById('summaryDeliveryList');
    list.innerHTML = "";
    
    // Filter history for significant reward events
    const significantSteps = stepHistory.filter(s => parseFloat(s.reward) > 0.05);
    if (significantSteps.length > 0) {
        significantSteps.forEach((d, i) => {
            const item = document.createElement('div');
            item.className = 'd-item';
            item.innerHTML = `<span>Event #${i+1}: ${d.message.split('!')[0]}</span> <span style="color:var(--green)">+${d.reward}</span>`;
            list.appendChild(item);
        });
    } else {
        list.innerHTML = `<div class="d-item" style="color:var(--dim)">No significant reward events recorded.</div>`;
    }

    modal.style.display = 'flex';
    
    // AUTO-ANALYSE: Trigger deep analysis on completion
    autoAnalyse();
}

async function autoAnalyse() {
    try {
        const res = await fetch(`/analyse/${currentTask}`);
        const data = await res.json();
        if (data && data.avg_reward) {
            // Update modal with analysis results if elements exist
            const avgEl = document.getElementById('summaryAvg');
            if (avgEl) {
                avgEl.innerHTML = `${data.avg_reward.toFixed(3)}`;
            }
            console.log("Auto-Analysis Complete:", data);
        }
    } catch(e) {
        console.warn("Auto-analysis failed (maybe no memory yet?):", e);
    }
}

function closeCompletionModal() {
    const modal = document.getElementById('completionModal');
    if (modal) modal.style.display = 'none';
}

function startNextTask() {
    closeCompletionModal();
    
    const sequence = {
        'easy_delivery': 'medium_delivery',
        'medium_delivery': 'hard_delivery',
        'hard_delivery': 'easy_delivery'
    };
    
    const nextTask = sequence[currentTask] || 'easy_delivery';
    currentTask = nextTask;
    
    // Update active state on buttons
    document.querySelectorAll('.task-btn').forEach(b => {
        b.classList.remove('active');
        if (b.dataset.task === nextTask) b.classList.add('active');
    });
    
    doReset();
    updateMissionLegend(); // Refresh legend
}

async function updateMissionLegend() {
    try {
        const res = await fetch('/tasks');
        const data = await res.json();
        const container = document.getElementById('legendTableContainer');
        if (!container || !data.tasks) return;
        
        // Ensure tasks are sorted Easy, Medium, Hard
        const order = ['easy_delivery', 'medium_delivery', 'hard_delivery'];
        const tasks = data.tasks.sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));
        
        container.innerHTML = `
            <table class="l-table">
                <thead>
                    <tr>
                        <th style="width: 25%">TECHNICAL METRIC</th>
                        <th style="width: 25%">EASY REWARD</th>
                        <th style="width: 25%">MEDIUM REWARD</th>
                        <th style="width: 25%">HARD REWARD</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="l-dim">Grid Resolution</td>
                        ${tasks.map(t => `<td class="l-hl">${t.width} x ${t.height}</td>`).join('')}
                    </tr>
                    <tr>
                        <td class="l-dim">Delivery Target</td>
                        ${tasks.map(t => `<td class="l-hl">+${t.r_delivery}</td>`).join('')}
                    </tr>
                    <tr>
                        <td class="l-dim">Package Count</td>
                        ${tasks.map(t => `<td class="l-hl">${t.n_deliveries}</td>`).join('')}
                    </tr>
                    <tr>
                        <td class="l-dim">Battery Capacity</td>
                        ${tasks.map(t => `<td class="l-hl">${t.battery_max}</td>`).join('')}
                    </tr>
                    <tr>
                        <td class="l-dim">Safe Flight Step</td>
                        ${tasks.map(t => `<td>+${t.r_step}</td>`).join('')}
                    </tr>
                    <tr>
                        <td class="l-dim">Collision Warning</td>
                        ${tasks.map(t => `<td>+${t.r_obstacle}</td>`).join('')}
                    </tr>
                    <tr>
                        <td class="l-dim">Critical Battery Fail</td>
                        ${tasks.map(t => `<td>+${t.r_battery_dead}</td>`).join('')}
                    </tr>
                     <tr>
                        <td class="l-dim">Restricted Airspace (Wall)</td>
                        ${tasks.map(t => `<td>+${t.r_wall}</td>`).join('')}
                    </tr>
                    <tr>
                        <td class="l-dim">Environment Density</td>
                        ${tasks.map(t => `<td class="l-dim">${t.n_buildings}B, ${t.n_trees}T, ${t.n_obstacles}O</td>`).join('')}
                    </tr>
                </tbody>
            </table>
        `;
    } catch(e) {
        console.warn("Could not update legend:", e);
    }
}

// Initial legend load
window.addEventListener('load', updateMissionLegend);
