let isAuto = false;
const btnStep = document.getElementById('btn-step');
const btnGrpo = document.getElementById('btn-grpo');
const btnAuto = document.getElementById('btn-auto');
const autoIndicator = document.getElementById('auto-indicator');
const autoStatusText = document.getElementById('auto-status-text');

const pstateVal = document.getElementById('pstate-val');
const clockVal = document.getElementById('clock-val');
const tempPowerVal = document.getElementById('temp-power-val');

const statRound = document.getElementById('stat-round');
const statGen = document.getElementById('stat-gen');

// Champion Elements
const champName = document.getElementById('champ-name');
const champAuthor = document.getElementById('champ-author');
const champGens = document.getElementById('champ-gens');
const champKills = document.getElementById('champ-kills');
const champLatency = document.getElementById('champ-latency');
const champRegs = document.getElementById('champ-regs');
const champUnroll = document.getElementById('champ-unroll');
const champCs = document.getElementById('champ-cs');
const champFma = document.getElementById('champ-fma');

// Challenger Elements
const challengerName = document.getElementById('challenger-name');
const challengerStrategy = document.getElementById('challenger-strategy');
const challengerCompile = document.getElementById('challenger-compile');
const challengerRegs = document.getElementById('challenger-regs');
const challengerLatency = document.getElementById('challenger-latency');
const challengerDelta = document.getElementById('challenger-delta');
const outcomeBadge = document.getElementById('outcome-badge');
const decisionText = document.getElementById('decision-text');
const statusOrb = document.getElementById('status-orb');

// Race Bars
const raceChampLabel = document.getElementById('race-champ-label');
const raceChampBar = document.getElementById('race-champ-bar');
const raceChallengerLabel = document.getElementById('race-challenger-label');
const raceChallengerBar = document.getElementById('race-challenger-bar');

// Tables & Logs
const hofTbody = document.getElementById('hof-tbody');
const logStream = document.getElementById('log-stream');

function updateUI(data) {
    if (!data) return;

    // Telemetry
    if (data.telemetry) {
        const t = data.telemetry;
        pstateVal.textContent = `${t.pstate} (Locked)`;
        clockVal.textContent = `${t.graphics_mhz} MHz`;
        tempPowerVal.textContent = `${t.temp_c}°C | ${t.power_w.toFixed(1)}W`;
    }

    statRound.textContent = data.round_num || 0;
    statGen.textContent = data.generation || 0;

    // Champion
    if (data.champion) {
        const c = data.champion;
        champName.textContent = c.name;
        champAuthor.textContent = c.author;
        champGens.textContent = c.generation;
        champKills.textContent = c.defeated_count;
        champLatency.innerHTML = `${c.latency_us.toFixed(2)} <small>µs</small>`;
        champRegs.textContent = c.registers;
        champUnroll.textContent = `x${c.unroll}`;
        champCs.textContent = c.cache_streaming ? "Enabled (ld.global.cs)" : "Standard";
        champFma.textContent = c.ptx_fma_mode ? "Inline PTX (fma.rn)" : "Standard C++";

        const baseUs = data.baseline_latency_us || 25.0;
        const statBaseEl = document.getElementById('stat-baseline');
        if (statBaseEl) statBaseEl.textContent = `${baseUs.toFixed(1)} µs`;

        raceChampLabel.textContent = `${c.name} (${c.latency_us.toFixed(2)} µs)`;
        const champPct = Math.min(100, Math.max(5, (c.latency_us / baseUs) * 100));
        raceChampBar.style.width = `${champPct}%`;
        raceChampBar.textContent = `${c.latency_us.toFixed(2)} µs (${(baseUs / c.latency_us).toFixed(1)}x faster)`;
    }

    // Challenger
    if (data.challenger) {
        const ch = data.challenger;
        const baseUs = data.baseline_latency_us || 25.0;
        challengerName.textContent = ch.name;
        challengerStrategy.textContent = ch.strategy;
        challengerCompile.textContent = `${ch.compile_time_ms.toFixed(0)} ms`;
        challengerRegs.textContent = ch.registers;
        challengerLatency.innerHTML = `${ch.latency_us.toFixed(2)} <small>µs</small>`;
        decisionText.textContent = ch.reason;

        outcomeBadge.className = "badge-outcome";
        if (ch.status === "ASCENDED") {
            outcomeBadge.classList.add("badge-ascended");
            outcomeBadge.textContent = "👑 ASCENDED (NEW CHAMP)";
            challengerDelta.className = "speedup-badge green";
            challengerDelta.textContent = `VICTORY (-${((data.champion.latency_us - ch.latency_us)/data.champion.latency_us * 100).toFixed(1)}%)`;
            statusOrb.textContent = "NEW CHAMPION!";
            statusOrb.style.color = "var(--neon-gold)";
        } else if (ch.status === "DEFEATED") {
            outcomeBadge.classList.add("badge-defeated");
            outcomeBadge.textContent = "💀 DEFEATED";
            challengerDelta.className = "speedup-badge red";
            challengerDelta.textContent = `LAG (+${((ch.latency_us - data.champion.latency_us)/data.champion.latency_us * 100).toFixed(1)}%)`;
            statusOrb.textContent = "CHAMP DEFENDED";
            statusOrb.style.color = "var(--neon-green)";
        } else {
            outcomeBadge.classList.add("badge-slain");
            outcomeBadge.textContent = "💥 SLAIN (CRASH)";
            challengerDelta.className = "speedup-badge red";
            challengerDelta.textContent = "FAILED";
            statusOrb.textContent = "MUTANT SLAIN";
            statusOrb.style.color = "var(--neon-red)";
        }

        raceChallengerLabel.textContent = `${ch.name} (${ch.latency_us < 990 ? ch.latency_us.toFixed(2) + ' µs' : 'CRASH'})`;
        const chPct = Math.min(100, Math.max(5, (ch.latency_us / baseUs) * 100));
        raceChallengerBar.style.width = ch.latency_us < 990 ? `${chPct}%` : '100%';
        raceChallengerBar.textContent = ch.latency_us < 990 ? `${ch.latency_us.toFixed(2)} µs` : 'FAILED / CRASHED';
    }

    // Hall of Fame
    if (data.hall_of_fame && data.hall_of_fame.length > 0) {
        hofTbody.innerHTML = data.hall_of_fame.map((item, idx) => `
            <tr>
                <td>#${idx + 1} ${idx === 0 ? '👑' : ''}</td>
                <td><strong>${item.name}</strong><br><small style="color: #64748b">${item.author || ''}</small></td>
                <td class="${idx === 0 ? 'gold' : 'cyan'}">${item.latency_us.toFixed(2)} µs</td>
                <td class="green">${item.speedup_vs_triton}</td>
                <td>${item.achieved_at || '--'}</td>
            </tr>
        `).join('');
    }

    // GRPO Cohort Grid
    const cohortPanel = document.getElementById('grpo-cohort-panel');
    const cohortContainer = document.getElementById('cohort-cards-container');
    if (data.last_grpo_cohort && data.last_grpo_cohort.length > 0) {
        cohortPanel.style.display = 'block';
        const bestAdv = Math.max(...data.last_grpo_cohort.map(c => c.grpo_advantage || 0));
        cohortContainer.innerHTML = data.last_grpo_cohort.map(c => {
            const isWin = (c.grpo_advantage === bestAdv);
            const advClass = (c.grpo_advantage >= 0) ? 'green' : 'red';
            return `
                <div class="cohort-card ${isWin ? 'winner' : ''}">
                    <div class="cohort-card-title ${isWin ? 'gold' : 'cyan'}">
                        ${c.name} ${isWin ? '👑 (VICTOR)' : ''}
                    </div>
                    <div class="cohort-card-meta">
                        Unroll: <strong>x${c.unroll}</strong> | CS: <strong>${c.cache_streaming ? 'ON' : 'OFF'}</strong> | PTX: <strong>${c.ptx_fma_mode ? 'ON' : 'OFF'}</strong>
                    </div>
                    <div class="cohort-card-meta">
                        Verify: ${c.verifiable_compile_pass ? '✅ Syntax' : '❌ Syntax'} | ${c.verifiable_math_pass ? '✅ Math' : '❌ Math'}
                    </div>
                    <div style="font-size: 1.1rem; font-weight: 800; margin: 0.2rem 0;" class="${isWin ? 'gold' : 'text-main'}">
                        ${c.latency_us < 990 ? c.latency_us.toFixed(2) + ' µs' : 'CRASH'}
                    </div>
                    <div class="cohort-advantage ${advClass}">
                        Advantage: ${c.grpo_advantage > 0 ? '+' : ''}${c.grpo_advantage ? c.grpo_advantage.toFixed(2) : '0.00'}
                    </div>
                </div>
            `;
        }).join('');
    }

    // Logs
    if (data.log && data.log.length > 0) {
        logStream.innerHTML = data.log.map(line => {
            let cls = "";
            if (line.includes("ASCENSION")) cls = "gold";
            else if (line.includes("SLAIN")) cls = "red";
            else if (line.includes("fell")) cls = "cyan";
            return `<div class="log-line ${cls}">${line}</div>`;
        }).join('');
    }

    // Auto status
    isAuto = !!data.auto_running;
    if (isAuto) {
        autoIndicator.classList.add("auto-running");
        autoStatusText.textContent = "AUTO-TOURNAMENT RUNNING";
        btnAuto.textContent = "⏹️ PAUSE TOURNAMENT";
        btnAuto.style.background = "rgba(255, 51, 102, 0.3)";
        btnAuto.style.borderColor = "var(--neon-red)";
    } else {
        autoIndicator.classList.remove("auto-running");
        autoStatusText.textContent = "MANUAL MODE";
        btnAuto.textContent = "⚡ START AUTO-TOURNAMENT";
        btnAuto.style.background = "rgba(181, 55, 242, 0.2)";
        btnAuto.style.borderColor = "var(--neon-purple)";
    }
}

// Fetch Initial Status
fetch('/api/status')
    .then(r => r.json())
    .then(updateUI)
    .catch(err => console.error("Initial load err:", err));

// Connect SSE stream
const evtSource = new EventSource('/api/stream');
evtSource.onmessage = function(e) {
    try {
        const data = JSON.parse(e.data);
        updateUI(data);
    } catch(err) {
        console.error("SSE parse error:", err);
    }
};

// Event Listeners
btnStep.addEventListener('click', () => {
    btnStep.disabled = true;
    btnStep.textContent = "COMPILING & BATTLING...";
    statusOrb.textContent = "COMPILING...";
    fetch('/api/step', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            updateUI(data);
            btnStep.disabled = false;
            btnStep.textContent = "⚔️ RUN ARENA MATCH";
        })
        .catch(err => {
            console.error("Step error:", err);
            btnStep.disabled = false;
            btnStep.textContent = "⚔️ RUN ARENA MATCH";
        });
});

btnGrpo.addEventListener('click', () => {
    btnGrpo.disabled = true;
    btnGrpo.textContent = "🧬 RUNNING RLVR / GRPO BATTLE (G=4)...";
    statusOrb.textContent = "GRPO BATTLE...";
    fetch('/api/grpo_step', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            btnGrpo.disabled = false;
            btnGrpo.textContent = "🧬 RLVR / GRPO BATTLE (G=4)";
            statusOrb.textContent = "GRPO DONE";
            // Refresh status
            fetch('/api/status').then(r => r.json()).then(updateUI);
        })
        .catch(err => {
            console.error("GRPO error:", err);
            btnGrpo.disabled = false;
            btnGrpo.textContent = "🧬 RLVR / GRPO BATTLE (G=4)";
        });
});

btnAuto.addEventListener('click', () => {
    if (isAuto) {
        fetch('/api/stop_auto', { method: 'POST' })
            .then(r => r.json())
            .then(() => {
                isAuto = false;
                autoIndicator.classList.remove("auto-running");
                autoStatusText.textContent = "MANUAL MODE";
                btnAuto.textContent = "⚡ START AUTO-TOURNAMENT";
            });
    } else {
        fetch('/api/start_auto', { method: 'POST' })
            .then(r => r.json())
            .then(() => {
                isAuto = true;
                autoIndicator.classList.add("auto-running");
                autoStatusText.textContent = "AUTO-TOURNAMENT RUNNING";
                btnAuto.textContent = "⏹️ PAUSE TOURNAMENT";
            });
    }
});

const workloadSelect = document.getElementById('workload-select');
if (workloadSelect) {
    workloadSelect.addEventListener('change', (e) => {
        const selected = e.target.value;
        statusOrb.textContent = "SWITCHING SCENARIO...";
        fetch('/api/set_workload', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ workload: selected })
        })
        .then(r => r.json())
        .then(() => {
            fetch('/api/status').then(r => r.json()).then(updateUI);
        });
    });
}
