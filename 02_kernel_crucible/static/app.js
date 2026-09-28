let isAuto = false;
let isUnslothPatched = false;

// Control Elements
const btnStep = document.getElementById('btn-step');
const btnGrpo = document.getElementById('btn-grpo');
const btnAuto = document.getElementById('btn-auto');
const btnUnslothToggle = document.getElementById('btn-unsloth-toggle');
const autoIndicator = document.getElementById('auto-indicator');
const autoStatusText = document.getElementById('auto-status-text');

// Telemetry Elements
const pstateVal = document.getElementById('pstate-val');
const clockVal = document.getElementById('clock-val');
const tempPowerVal = document.getElementById('temp-power-val');
const statRound = document.getElementById('stat-round');
const statGen = document.getElementById('stat-gen');
const statBaseline = document.getElementById('stat-baseline');

// Unsloth Elements
const unslothHeaderBadge = document.getElementById('unsloth-header-badge');
const unslothStatusTag = document.getElementById('unsloth-status-tag');
const unslothSpeedupTag = document.getElementById('unsloth-speedup-tag');

// Champion Elements
const champName = document.getElementById('champ-name');
const champAuthor = document.getElementById('champ-author');
const champGens = document.getElementById('champ-gens');
const champKills = document.getElementById('champ-kills');
const champLatency = document.getElementById('champ-latency');
const champSpeedupBadge = document.getElementById('champ-speedup-badge');
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
const raceBaselineLabel = document.getElementById('race-baseline-label');
const raceBaselineBar = document.getElementById('race-baseline-bar');
const raceTorchBar = document.getElementById('race-torch-bar');
const raceChampLabel = document.getElementById('race-champ-label');
const raceChampBar = document.getElementById('race-champ-bar');
const raceChallengerLabel = document.getElementById('race-challenger-label');
const raceChallengerBar = document.getElementById('race-challenger-bar');

// Tables & Logs
const hofTbody = document.getElementById('hof-tbody');
const logStream = document.getElementById('log-stream');
const workloadSelect = document.getElementById('workload-select');

// Track state for animations
let prevChampName = "";
let prevChallengerName = "";
let prevChampLat = 0;
let prevChallengerLat = 0;

/* ==========================================================================
   REACTBITS MICRO-INTERACTION ENGINES (https://reactbits.dev)
   ========================================================================== */

// 1. ReactBits DecryptedText Engine
const GLYPHS = "!@#$%^&*()_+-=<>?/\\[]{}0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
function scrambleDecryptedText(element, finalStr, duration = 400) {
    if (!element || !finalStr || element.textContent === finalStr) return;
    element.classList.add("decrypted-active");
    const startTime = performance.now();
    const length = finalStr.length;

    function frame(now) {
        const elapsed = now - startTime;
        const progress = Math.min(1, elapsed / duration);
        const resolvedChars = Math.floor(progress * length);

        let result = "";
        for (let i = 0; i < length; i++) {
            if (i < resolvedChars) {
                result += finalStr[i];
            } else {
                result += GLYPHS[Math.floor(Math.random() * GLYPHS.length)];
            }
        }
        element.textContent = result;

        if (progress < 1) {
            requestAnimationFrame(frame);
        } else {
            element.textContent = finalStr;
            element.classList.remove("decrypted-active");
        }
    }
    requestAnimationFrame(frame);
}

// 2. ReactBits NumberTicker / CountUp Engine
function animateNumber(element, start, end, decimals = 2, suffix = "") {
    if (!element || start === end) {
        if (element) element.innerHTML = `${end.toFixed(decimals)} <small>${suffix}</small>`;
        return;
    }
    const startTime = performance.now();
    const duration = 450;

    function step(now) {
        const progress = Math.min(1, (now - startTime) / duration);
        const ease = 1 - Math.pow(1 - progress, 3); // Ease-out cubic
        const current = start + (end - start) * ease;
        element.innerHTML = `${current.toFixed(decimals)} <small>${suffix}</small>`;

        if (progress < 1) {
            requestAnimationFrame(step);
        } else {
            element.innerHTML = `${end.toFixed(decimals)} <small>${suffix}</small>`;
        }
    }
    requestAnimationFrame(step);
}

// 3. ReactBits Spotlight Card Tracker
function initReactBitsSpotlight() {
    document.querySelectorAll('.spotlight-card').forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            card.style.setProperty('--mouse-x', `${x}px`);
            card.style.setProperty('--mouse-y', `${y}px`);
        });
    });
}

// 4. ReactBits Interactive Canvas Particles Engine
function initReactBitsParticles() {
    const canvas = document.getElementById('particles-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    window.addEventListener('resize', () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    });

    const particles = [];
    const count = 45;
    for (let i = 0; i < count; i++) {
        particles.push({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.4,
            vy: (Math.random() - 0.5) * 0.4,
            radius: Math.random() * 1.5 + 1
        });
    }

    let mouseX = -1000, mouseY = -1000;
    window.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });

    function loop() {
        ctx.clearRect(0, 0, width, height);

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0) p.x = width;
            if (p.x > width) p.x = 0;
            if (p.y < 0) p.y = height;
            if (p.y > height) p.y = 0;

            // Draw particle
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(0, 240, 255, 0.4)";
            ctx.fill();

            // Connect nearby particles
            for (let j = i + 1; j < particles.length; j++) {
                const p2 = particles[j];
                const dx = p.x - p2.x;
                const dy = p.y - p2.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 110) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(0, 240, 255, ${0.15 * (1 - dist / 110)})`;
                    ctx.lineWidth = 0.6;
                    ctx.stroke();
                }
            }

            // Connect to mouse
            const mdx = p.x - mouseX;
            const mdy = p.y - mouseY;
            const mdist = Math.sqrt(mdx * mdx + mdy * mdy);
            if (mdist < 140) {
                ctx.beginPath();
                ctx.moveTo(p.x, p.y);
                ctx.lineTo(mouseX, mouseY);
                ctx.strokeStyle = `rgba(181, 55, 242, ${0.25 * (1 - mdist / 140)})`;
                ctx.lineWidth = 0.8;
                ctx.stroke();
            }
        }
        requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
}

/* ==========================================================================
   UI DATA BINDING & UPDATER
   ========================================================================== */

function updateUI(data) {
    if (!data) return;

    // 1. Hardware Telemetry
    if (data.telemetry) {
        const t = data.telemetry;
        if (pstateVal) pstateVal.textContent = `${t.pstate} (Locked)`;
        if (clockVal) clockVal.textContent = `${t.graphics_mhz} MHz`;
        if (tempPowerVal) tempPowerVal.textContent = `${t.temp_c}°C | ${t.power_w.toFixed(1)}W`;
    }

    if (statRound) statRound.textContent = data.round_num || 0;
    if (statGen) statGen.textContent = data.generation || 0;

    const baseUs = data.baseline_latency_us || 28.29;
    if (statBaseline) statBaseline.textContent = `${baseUs.toFixed(2)} µs`;

    // 2. Unsloth Monkey-Patch Status
    isUnslothPatched = !!data.unsloth_patched;
    if (unslothStatusTag && unslothSpeedupTag && btnUnslothToggle) {
        if (isUnslothPatched) {
            unslothStatusTag.className = "status-tag active";
            unslothStatusTag.textContent = "STATUS: ACTIVE (CRUCIBLE KERNEL)";
            unslothHeaderBadge.textContent = "Patched (Crucible SM86)";
            unslothHeaderBadge.className = "val green";
            btnUnslothToggle.textContent = "🔄 REVERT TO PYTORCH NATIVE";
            btnUnslothToggle.className = "btn btn-unsloth btn-revert";
        } else {
            unslothStatusTag.className = "status-tag";
            unslothStatusTag.textContent = "STATUS: NATIVE PYTORCH";
            unslothHeaderBadge.textContent = "Online (Port 8888)";
            unslothHeaderBadge.className = "val purple";
            btnUnslothToggle.textContent = "🦥 INJECT INTO UNSLOTH & PYTORCH";
            btnUnslothToggle.className = "btn btn-unsloth";
        }
    }

    // 3. Champion Card & Metrics
    if (data.champion) {
        const c = data.champion;
        if (champName) {
            if (c.name !== prevChampName) {
                scrambleDecryptedText(champName, c.name, 450);
                prevChampName = c.name;
            }
        }
        if (champAuthor) champAuthor.textContent = c.author;
        if (champGens) champGens.textContent = c.generation;
        if (champKills) champKills.textContent = c.defeated_count;

        if (champLatency) {
            animateNumber(champLatency, prevChampLat || c.latency_us, c.latency_us, 2, "µs");
            prevChampLat = c.latency_us;
        }

        if (champRegs) champRegs.textContent = c.registers;
        if (champUnroll) champUnroll.textContent = `x${c.unroll}`;
        if (champCs) champCs.textContent = c.cache_streaming ? "Enabled (ld.global.cs)" : "Standard";
        if (champFma) champFma.textContent = c.ptx_fma_mode ? "Inline PTX (fma.rn)" : "Standard C++";

        const speedupVsBase = (baseUs / c.latency_us);
        if (champSpeedupBadge) {
            champSpeedupBadge.textContent = `${speedupVsBase.toFixed(2)}x Faster than Baseline`;
        }
        if (unslothSpeedupTag) {
            unslothSpeedupTag.textContent = `CRUCIBLE SPEEDUP: ${speedupVsBase.toFixed(2)}x SILICON`;
        }

        // Live Race Bars
        if (raceBaselineBar) {
            raceBaselineBar.textContent = `${baseUs.toFixed(2)} µs (Unoptimized Baseline)`;
            raceBaselineBar.style.width = "100%";
        }
        if (raceTorchBar) {
            const torchEstUs = baseUs * 0.72; // Typical PyTorch Cutlass SDPA
            const torchPct = Math.min(100, Math.max(10, (torchEstUs / baseUs) * 100));
            raceTorchBar.style.width = `${torchPct}%`;
            raceTorchBar.textContent = `${torchEstUs.toFixed(2)} µs (PyTorch SDPA)`;
        }
        if (raceChampBar && raceChampLabel) {
            raceChampLabel.textContent = `${c.name} (${c.latency_us.toFixed(2)} µs)`;
            const champPct = Math.min(100, Math.max(5, (c.latency_us / baseUs) * 100));
            raceChampBar.style.width = `${champPct}%`;
            raceChampBar.textContent = `${c.latency_us.toFixed(2)} µs (${speedupVsBase.toFixed(2)}x faster)`;
        }
    }

    // 4. Challenger Card & Metrics
    if (data.challenger) {
        const ch = data.challenger;
        if (challengerName) {
            if (ch.name !== prevChallengerName) {
                scrambleDecryptedText(challengerName, ch.name, 400);
                prevChallengerName = ch.name;
            }
        }
        if (challengerStrategy) challengerStrategy.textContent = ch.strategy;
        if (challengerCompile) challengerCompile.textContent = `${ch.compile_time_ms.toFixed(0)} ms`;
        if (challengerRegs) challengerRegs.textContent = ch.registers;
        
        if (challengerLatency) {
            if (ch.latency_us < 990) {
                animateNumber(challengerLatency, prevChallengerLat || ch.latency_us, ch.latency_us, 2, "µs");
                prevChallengerLat = ch.latency_us;
            } else {
                challengerLatency.innerHTML = `CRASH <small>µs</small>`;
            }
        }

        if (decisionText) decisionText.textContent = ch.reason;

        if (outcomeBadge && challengerDelta && statusOrb) {
            outcomeBadge.className = "badge-outcome";
            if (ch.status === "ASCENDED") {
                outcomeBadge.classList.add("badge-ascended");
                outcomeBadge.textContent = "👑 ASCENDED (NEW CHAMP)";
                challengerDelta.className = "speedup-badge green";
                const diffPct = ((data.champion.latency_us - ch.latency_us) / data.champion.latency_us * 100).toFixed(1);
                challengerDelta.textContent = `VICTORY (-${diffPct}%)`;
                statusOrb.textContent = "NEW CHAMPION!";
                statusOrb.style.color = "var(--neon-gold)";
            } else if (ch.status === "DEFEATED") {
                outcomeBadge.classList.add("badge-defeated");
                outcomeBadge.textContent = "💀 DEFEATED";
                challengerDelta.className = "speedup-badge red";
                const diffPct = ((ch.latency_us - data.champion.latency_us) / data.champion.latency_us * 100).toFixed(1);
                challengerDelta.textContent = `LAG (+${diffPct}%)`;
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
        }

        if (raceChallengerLabel && raceChallengerBar) {
            raceChallengerLabel.textContent = `${ch.name} (${ch.latency_us < 990 ? ch.latency_us.toFixed(2) + ' µs' : 'CRASH'})`;
            const chPct = Math.min(100, Math.max(5, (ch.latency_us / baseUs) * 100));
            raceChallengerBar.style.width = ch.latency_us < 990 ? `${chPct}%` : '100%';
            raceChallengerBar.textContent = ch.latency_us < 990 ? `${ch.latency_us.toFixed(2)} µs` : 'FAILED / CRASHED';
        }
    }

    // 5. Hall of Fame
    if (data.hall_of_fame && data.hall_of_fame.length > 0 && hofTbody) {
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

    // 6. GRPO Cohort Grid
    const cohortPanel = document.getElementById('grpo-cohort-panel');
    const cohortContainer = document.getElementById('cohort-cards-container');
    if (cohortPanel && cohortContainer) {
        if (data.last_grpo_cohort && data.last_grpo_cohort.length > 0) {
            cohortPanel.style.display = 'block';
            const bestAdv = Math.max(...data.last_grpo_cohort.map(c => c.grpo_advantage || 0));
            cohortContainer.innerHTML = data.last_grpo_cohort.map(c => {
                const isWin = (c.grpo_advantage === bestAdv);
                const advClass = (c.grpo_advantage >= 0) ? 'green' : 'red';
                return `
                    <div class="cohort-card spotlight-card ${isWin ? 'winner laser-border' : ''}">
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
            initReactBitsSpotlight();
        }
    }

    // 7. Live Combat Log
    if (data.log && data.log.length > 0 && logStream) {
        logStream.innerHTML = data.log.map(line => {
            let cls = "";
            if (line.includes("ASCENSION")) cls = "gold";
            else if (line.includes("SLAIN")) cls = "red";
            else if (line.includes("fell")) cls = "cyan";
            return `<div class="log-line ${cls}">${line}</div>`;
        }).join('');
    }

    // 8. Auto Status
    isAuto = !!data.auto_running;
    if (autoIndicator && autoStatusText && btnAuto) {
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
}

// Initial Bootstrapping
initReactBitsSpotlight();
initReactBitsParticles();

fetch('/api/status')
    .then(r => r.json())
    .then(updateUI)
    .catch(err => console.error("Initial load err:", err));

// SSE Stream Connection
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
if (btnStep) {
    btnStep.addEventListener('click', () => {
        btnStep.disabled = true;
        btnStep.textContent = "COMPILING & BATTLING...";
        if (statusOrb) statusOrb.textContent = "COMPILING...";
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
}

if (btnGrpo) {
    btnGrpo.addEventListener('click', () => {
        btnGrpo.disabled = true;
        btnGrpo.textContent = "🧬 RLVR BATTLE (G=4)...";
        if (statusOrb) statusOrb.textContent = "GRPO BATTLE...";
        fetch('/api/grpo_step', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                btnGrpo.disabled = false;
                btnGrpo.textContent = "🧬 RLVR / GRPO BATTLE (G=4)";
                if (statusOrb) statusOrb.textContent = "GRPO DONE";
                fetch('/api/status').then(r => r.json()).then(updateUI);
            })
            .catch(err => {
                console.error("GRPO error:", err);
                btnGrpo.disabled = false;
                btnGrpo.textContent = "🧬 RLVR / GRPO BATTLE (G=4)";
            });
    });
}

if (btnAuto) {
    btnAuto.addEventListener('click', () => {
        if (isAuto) {
            fetch('/api/stop_auto', { method: 'POST' })
                .then(r => r.json())
                .then(() => {
                    isAuto = false;
                    fetch('/api/status').then(r => r.json()).then(updateUI);
                });
        } else {
            fetch('/api/start_auto', { method: 'POST' })
                .then(r => r.json())
                .then(() => {
                    isAuto = true;
                    fetch('/api/status').then(r => r.json()).then(updateUI);
                });
        }
    });
}

if (btnUnslothToggle) {
    btnUnslothToggle.addEventListener('click', () => {
        btnUnslothToggle.disabled = true;
        const endpoint = isUnslothPatched ? '/api/unsloth_unpatch' : '/api/unsloth_patch';
        btnUnslothToggle.textContent = isUnslothPatched ? "REVERTING TO NATIVE..." : "INJECTING KERNEL...";
        
        fetch(endpoint, { method: 'POST' })
            .then(r => r.json())
            .then(() => {
                btnUnslothToggle.disabled = false;
                fetch('/api/status').then(r => r.json()).then(updateUI);
            })
            .catch(err => {
                console.error("Unsloth patch error:", err);
                btnUnslothToggle.disabled = false;
            });
    });
}

if (workloadSelect) {
    workloadSelect.addEventListener('change', (e) => {
        const selected = e.target.value;
        if (statusOrb) statusOrb.textContent = "SWITCHING SCENARIO...";
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
