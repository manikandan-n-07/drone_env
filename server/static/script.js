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
let currentTask = 'graders:grade_easy'; 
let autoTimer   = null;
let logTimer    = null;
let obs         = null;
let sessionId   = Math.random().toString(36).substring(7);
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
    renderFleet(obs);

    // Primary drone info for telemetry (Drone 0)
    const primary = obs.drones[0] || { x: 0, y: 0, battery: 0 };

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
            if (primary.battery < 0.2) {
                batEl.classList.add('critical-flash');
            } else {
                batEl.classList.remove('critical-flash');
            }
        }
    }
    
    // Battery (Primary Drone)
    const batPct = Math.max(0, Math.round(primary.battery * 100));
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
        drone_0_pos: `(${primary.x}, ${primary.y})`,
        reward: obs.reward_last.toFixed(4),
        total_reward: obs.reward_total.toFixed(4),
        score: obs.score.toFixed(4),
        battery_0: batPct,
        message: obs.message
    });

    // Message bar
    const msgEl = document.getElementById('msgBar');
    msgEl.textContent = obs.message;
    msgEl.className = 'msg-bar' + (obs.done ? (obs.deliveries_done === obs.deliveries_total ? ' good' : ' bad') : '');

    // Subtitles (Mission Chatter Scrolling)
    if (obs.message) {
        // Split combined messages (separated by |) and add them individually
        const parts = obs.message.split(' | ');
        parts.forEach(p => addSubtitle(p));
    }

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

function renderFleet(obs) {
    const fleetGrid = document.getElementById('fleetGrid');
    const droneCount = document.getElementById('droneCount');
    if (!fleetGrid) return;

    droneCount.textContent = `${obs.drones.length} Drones`;
    
    let html = '';
    obs.drones.forEach(d => {
        const isIdle = d.target_id === null;
        const status = d.has_package ? 'Returning (Done ✅)' : (isIdle ? 'Idle' : 'To Package');
        const statusClass = d.has_package ? 'warn' : (isIdle ? 'dim' : 'success');
        const batColor = d.battery > 0.5 ? 'var(--green)' : (d.battery > 0.2 ? 'var(--amber)' : 'var(--red)');

        html += `
            <div class="drone-card ${isIdle ? '' : 'active'}">
                <div class="drone-card-indicator"></div>
                <div class="drone-card-header">
                    <span class="drone-id">🚁 DRONE-${d.id}</span>
                    <span class="drone-status-tag" style="color:var(--${statusClass})">${status}</span>
                </div>
                <div class="drone-card-body">
                    <div class="drone-stat-row">
                        <span class="drone-stat-label">Position</span>
                        <span class="drone-stat-value">(${d.x}, ${d.y})</span>
                    </div>
                    <div class="drone-stat-row">
                        <span class="drone-stat-label">Target ID</span>
                        <span class="drone-stat-value">${d.target_id !== null ? d.target_id : '—'}</span>
                    </div>
                    <div class="drone-stat-row">
                        <span class="drone-stat-label">Battery</span>
                        <span class="drone-stat-value">${Math.round(d.battery * 100)}%</span>
                    </div>
                    <div class="drone-battery-mini">
                        <div class="drone-battery-mini-fill" style="width: ${d.battery * 100}%; background: ${batColor}"></div>
                    </div>
                </div>
            </div>
        `;
    });
    fleetGrid.innerHTML = html;
}

function updateLiveTelemetry(obs) {
    const consoleEl = document.getElementById('memoryConsole');
    if (!consoleEl) return;
    
    const primary = obs.drones[0] || { x: 0, y: 0, battery: 0 };
    const telemetry = {
        step: obs.step_count,
        active_drones: obs.drones.length,
        primary_pos: `(${primary.x}, ${primary.y})`,
        reward: parseFloat(obs.reward_last.toFixed(4)),
        total_reward: parseFloat(obs.reward_total.toFixed(4)),
        primary_battery: `${Math.round(primary.battery * 100)}%`,
        status: obs.message
    };
    
    // Append JSON line
    const line = document.createElement('div');
    line.className = 'console-line';
    line.style.color = '#00e5ff'; // Cyan for JSON
    line.innerHTML = `> ${JSON.stringify(telemetry)}`;
    
    consoleEl.appendChild(line);
    consoleEl.scrollTop = consoleEl.scrollHeight;
    
    // Auto-prune
    if (consoleEl.children.length > 50) {
        consoleEl.removeChild(consoleEl.firstChild);
    }
}

function addSubtitle(text) {
    const container = document.getElementById('subListContainer');
    if (!container) return;

    // Check if duplicate of last message to avoid spam
    if (container.lastElementChild && container.lastElementChild.querySelector('.sub-text').textContent === text) {
        return;
    }

    const line = document.createElement('div');
    line.className = 'sub-line';
    
    // Parse Identity from message if possible (e.g., "🚁 Drone 0: Message")
    let prefix = "SYSTEM";
    let displayText = text;
    
    if (text.includes(':')) {
        const parts = text.split(':');
        prefix = parts[0].trim().replace('🚁 ', '').replace('🛰️ ', '').replace('✅ ', '');
        displayText = parts.slice(1).join(':').trim();
    } else {
        prefix = "CORE_AI";
    }

    line.innerHTML = `
        <span class="sub-prefix">${prefix.toUpperCase()}:</span>
        <span class="sub-text">${displayText}</span>
    `;

    container.appendChild(line);

    // Keep only last 4 messages for a clean scrolling look
    if (container.children.length > 4) {
        container.removeChild(container.firstChild);
    }
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
            body: JSON.stringify({ task_name: currentTask, session_id: sessionId })
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
        // Manual control: Send same direction to all drones or just primary?
        // Let's send it to all drones for simplicity in manual mode
        const actions = {};
        obs.drones.forEach(d => { actions[d.id] = dir; });
        
        const r = await fetch(`${BASE}/step`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ actions: actions, session_id: sessionId })
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
        const r = await fetch(`${BASE}/analyse/${encodeURIComponent(currentTask)}`);
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
        const res = await r.json();
        const droneActions = res.actions || {};
        if (autoActive) {
            const rStep = await fetch(`${BASE}/step`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ actions: droneActions, session_id: sessionId })
            });
            const data = await rStep.json();
            updateUI(data);
            setTimeout(autoLoop, 800);
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
        const res = await fetch(`/analyse/${encodeURIComponent(currentTask)}`);
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
        'graders:grade_easy': 'graders:grade_medium',
        'graders:grade_medium': 'graders:grade_hard',
        'graders:grade_hard': 'graders:grade_easy'
    };
    
    const nextTask = sequence[currentTask] || 'drone_env.graders.easy:grade_easy';
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
        const order = ['graders:grade_easy', 'graders:grade_medium', 'graders:grade_hard'];
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
