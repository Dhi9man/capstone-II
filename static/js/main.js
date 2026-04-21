/* ============================================================
   LipSyncD — Frontend JS
   ============================================================ */

let selectedFile = null;
let timelineChart = null;
let demoMode = false;

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initWaveCanvas();
  initDragDrop();
  document.getElementById('videoFile').addEventListener('change', handleFileSelect);
  fetchModelStatus();
});

async function fetchModelStatus() {
  try {
    const res  = await fetch('/model-status');
    const data = await res.json();
    const banner = document.getElementById('modelBanner');
    const text   = document.getElementById('modelBannerText');
    if (data.trained) {
      banner.className = 'model-banner trained';
      const i = data.info;
      text.innerHTML = `✓ Trained model loaded &nbsp;·&nbsp; Val Acc <strong>${i.val_acc}%</strong> &nbsp;·&nbsp; AUC <strong>${i.val_auc}</strong> &nbsp;·&nbsp; Epoch ${i.epoch} &nbsp;·&nbsp; <em>${(i.manips||[]).join(', ')}</em>`;
    } else {
      banner.className = 'model-banner untrained';
      text.innerHTML = '⚠ No trained weights found — using heuristic mode. Run <code>python train.py --data /path/to/faceforensics</code> to train.';
    }
  } catch(e) {
    document.getElementById('modelBannerText').textContent = 'Could not check model status.';
  }
}

// ── Animated background waveform ─────────────────────────────
function initWaveCanvas() {
  const canvas = document.getElementById('waveCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  let t = 0;
  const waves = [
    { amp: 0.04, freq: 0.008, speed: 0.3, color: 'rgba(0,212,255,0.15)' },
    { amp: 0.025, freq: 0.013, speed: 0.5, color: 'rgba(0,212,255,0.08)' },
    { amp: 0.015, freq: 0.02,  speed: 0.7, color: 'rgba(0,150,199,0.12)' },
  ];

  function draw() {
    const { width: W, height: H } = canvas;
    ctx.clearRect(0, 0, W, H);

    waves.forEach(wave => {
      ctx.beginPath();
      for (let x = 0; x <= W; x++) {
        const y = H * 0.5 + Math.sin(x * wave.freq + t * wave.speed) * H * wave.amp
                           + Math.sin(x * wave.freq * 1.7 + t * wave.speed * 0.6) * H * wave.amp * 0.5;
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = wave.color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    t += 0.5;
    requestAnimationFrame(draw);
  }
  draw();
}

// ── Drag & drop ───────────────────────────────────────────────
function initDragDrop() {
  const zone = document.getElementById('uploadZone');
  if (!zone) return;
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files.length > 0) setFile(files[0]);
  });
  zone.addEventListener('click', () => document.getElementById('videoFile').click());
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) setFile(file);
}

function setFile(file) {
  const allowed = ['video/mp4','video/avi','video/quicktime','video/x-matroska','video/webm','video/x-msvideo'];
  if (!allowed.some(t => file.type.startsWith('video')) && !file.name.match(/\.(mp4|avi|mov|mkv|webm)$/i)) {
    alert('Please upload a video file (mp4, avi, mov, mkv, webm)');
    return;
  }
  selectedFile = file;
  demoMode = false;
  document.getElementById('uploadZone').style.display = 'none';
  document.getElementById('filePreview').style.display = 'flex';
  document.getElementById('fileName').textContent = file.name;
  document.getElementById('fileSize').textContent = formatSize(file.size);
  document.getElementById('analyzeBtn').disabled = false;
}

function clearFile() {
  selectedFile = null;
  demoMode = false;
  document.getElementById('uploadZone').style.display = 'block';
  document.getElementById('filePreview').style.display = 'none';
  document.getElementById('videoFile').value = '';
  document.getElementById('analyzeBtn').disabled = true;
}

// ── Demo mode ─────────────────────────────────────────────────
function runDemo(type) {
  demoMode = type;
  selectedFile = null;
  const labels = { deepfake: '🎭 Demo: Deepfake', authentic: '✅ Demo: Authentic', lipsync: '◎ Demo: Lip-Sync' };
  document.getElementById('uploadZone').style.display = 'none';
  document.getElementById('filePreview').style.display = 'flex';
  document.getElementById('fileName').textContent = labels[type] || type;
  document.getElementById('fileSize').textContent = 'Synthetic demo — no real video';
  document.getElementById('analyzeBtn').disabled = false;
}

// ── Main analysis ─────────────────────────────────────────────
async function analyzeVideo() {
  showProgress();

  const steps = document.querySelectorAll('.progress-steps .step');
  let stepIndex = 0;
  const total = steps.length;

  const tick = setInterval(() => {
    if (stepIndex > 0) steps[stepIndex - 1].classList.replace('active','done');
    if (stepIndex < total) {
      steps[stepIndex].classList.add('active');
      const pct = Math.round(((stepIndex + 1) / total) * 90);
      document.getElementById('progressBar').style.width = pct + '%';
      document.getElementById('progressLabel').textContent = steps[stepIndex].querySelector('.step-text').textContent;
      stepIndex++;
    }
  }, demoMode ? 350 : 600);

  try {
    let data;
    if (demoMode) {
      await sleep(total * 350 + 400);
      const res = await fetch('/demo', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ type: demoMode })
      });
      data = await res.json();
    } else {
      const formData = new FormData();
      formData.append('video', selectedFile);
      const res = await fetch('/analyze', { method: 'POST', body: formData });
      data = await res.json();
    }

    clearInterval(tick);
    steps.forEach(s => s.classList.replace('active','done'));
    document.getElementById('progressBar').style.width = '100%';
    document.getElementById('progressLabel').textContent = 'Analysis complete!';

    await sleep(400);
    showResults(data);
  } catch (err) {
    clearInterval(tick);
    alert('Analysis failed: ' + err.message);
    resetUI();
  }
}

// ── Show progress panel ───────────────────────────────────────
function showProgress() {
  document.getElementById('uploadPanel').style.display = 'none';
  document.getElementById('progressPanel').style.display = 'block';
  document.getElementById('resultsPanel').style.display = 'none';
  document.querySelectorAll('.step').forEach(s => {
    s.classList.remove('active','done');
  });
  document.getElementById('progressBar').style.width = '0%';
}

// ── Render results ────────────────────────────────────────────
function showResults(data) {
  if (data.error) { alert('Error: ' + data.error); resetUI(); return; }

  document.getElementById('progressPanel').style.display = 'none';
  document.getElementById('resultsPanel').style.display = 'block';

  const isFake = data.verdict === 'DEEPFAKE';
  const score  = data.final_score;

  // Job ID
  document.getElementById('jobId').textContent = '#' + data.job_id + (data.demo ? ' · DEMO' : '');

  // Verdict card
  const card = document.getElementById('verdictCard');
  card.className = 'verdict-card ' + (isFake ? 'fake' : 'real');
  document.getElementById('verdictIcon').textContent = isFake ? '⚠' : '✓';
  document.getElementById('verdictText').textContent = data.verdict;
  const riskEl = document.getElementById('verdictRisk');
  riskEl.textContent = data.risk_level + ' RISK';
  riskEl.style.cssText = `background: ${riskColor(data.risk_level, 0.1)}; color: ${riskColor(data.risk_level, 1)}; border: 1px solid ${riskColor(data.risk_level, 0.3)};`;

  // Gauge
  drawGauge(score);
  document.getElementById('gaugeScore').textContent = Math.round(score * 100) + '%';
  document.getElementById('gaugeScore').style.color = isFake ? 'var(--danger)' : 'var(--safe)';

  // Score bars
  const scoreMap = {
    'sc-sync':     ['Sync Score',     data.scores.sync_score],
    'sc-artifact': ['Artifact Score', data.scores.artifact_score],
    'sc-temporal': ['Temporal Score', data.scores.temporal_score],
    'sc-model':    ['Model Score',    data.scores.model_score],
  };
  Object.entries(scoreMap).forEach(([id, [label, val]]) => {
    const card = document.getElementById(id);
    const bar  = card.querySelector('.sc-bar-inner');
    const valEl = card.querySelector('.sc-value');
    const pct = Math.round(val * 100);
    bar.style.width = pct + '%';
    bar.style.background = scoreGradient(val);
    bar.style.boxShadow  = `0 0 8px ${scoreGradient(val)}`;
    valEl.textContent = pct + '%';
    valEl.style.color = scoreGradient(val);
  });

  // Timeline chart
  renderTimeline(data.timeline);

  // MFCC heatmap
  renderMFCC(data.mfcc_heatmap);

  // Sync details
  const sd = data.sync_details || {};
  const sdEl = document.getElementById('syncDetails');
  sdEl.innerHTML = Object.entries(sd).map(([k, v]) =>
    `<div class="sd-item"><div class="sd-label">${k.replace(/_/g,' ')}</div><div class="sd-value">${v}</div></div>`
  ).join('');

  // Library status
  const libs = data.libraries || {};
  document.getElementById('libsStatus').innerHTML =
    Object.entries(libs).map(([lib, ok]) =>
      `<span class="lib-pill ${ok?'ok':'missing'}">${ok?'✓':'✗'} ${lib}</span>`
    ).join('');

  // Animate in
  document.getElementById('resultsPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── Gauge canvas ──────────────────────────────────────────────
function drawGauge(score) {
  const canvas = document.getElementById('gaugeCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H * 0.85;
  const r = Math.min(W, H * 1.6) * 0.42;
  const startAngle = Math.PI;
  const endAngle   = 2 * Math.PI;
  const valueAngle = startAngle + score * Math.PI;

  // Track
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, endAngle);
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 8;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Fill
  const grad = ctx.createLinearGradient(cx - r, 0, cx + r, 0);
  grad.addColorStop(0,   '#00e676');
  grad.addColorStop(0.5, '#ffb800');
  grad.addColorStop(1,   '#ff3d5a');
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, valueAngle);
  ctx.strokeStyle = grad;
  ctx.lineWidth = 8;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Needle
  const nx = cx + (r * 0.75) * Math.cos(valueAngle);
  const ny = cy + (r * 0.75) * Math.sin(valueAngle);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(nx, ny);
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
  ctx.fillStyle = '#fff';
  ctx.fill();
}

// ── Timeline Chart ────────────────────────────────────────────
function renderTimeline(data) {
  if (timelineChart) { timelineChart.destroy(); timelineChart = null; }
  const ctx = document.getElementById('timelineChart').getContext('2d');
  const labels = data.map((_, i) => `${Math.round(i * 100 / data.length)}%`);

  timelineChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data,
        borderColor: 'rgba(0,212,255,0.8)',
        backgroundColor: ctx => {
          const g = ctx.chart.ctx.createLinearGradient(0,0,0,120);
          g.addColorStop(0, 'rgba(0,212,255,0.2)');
          g.addColorStop(1, 'rgba(0,212,255,0.0)');
          return g;
        },
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointRadius: 0,
        pointHoverRadius: 4,
      }]
    },
    options: {
      responsive: true,
      animation: { duration: 1200 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `Prob: ${Math.round(ctx.raw * 100)}%`
          }
        }
      },
      scales: {
        y: {
          min: 0, max: 1,
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: {
            color: '#4a6880', font: { family: "'Space Mono'" },
            callback: v => Math.round(v * 100) + '%'
          }
        },
        x: {
          grid: { display: false },
          ticks: { color: '#4a6880', font: { family: "'Space Mono'", size: 10 }, maxTicksLimit: 8 }
        }
      }
    }
  });
}

// ── MFCC Heatmap ──────────────────────────────────────────────
function renderMFCC(data) {
  const canvas = document.getElementById('mfccCanvas');
  canvas.width  = canvas.offsetWidth || 700;
  canvas.height = 130;
  const ctx = canvas.getContext('2d');
  const rows = data.length, cols = data[0].length;
  const cw = canvas.width / cols, ch = canvas.height / rows;

  // Compute min/max for normalization
  let mn = Infinity, mx = -Infinity;
  data.forEach(row => row.forEach(v => { if(v<mn)mn=v; if(v>mx)mx=v; }));

  data.forEach((row, r) => {
    row.forEach((val, c) => {
      const t = (val - mn) / (mx - mn + 1e-8);
      ctx.fillStyle = mfccColor(t);
      ctx.fillRect(c * cw, r * ch, cw, ch);
    });
  });

  // Row labels
  ctx.font = "9px 'Space Mono'";
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  for (let r = 0; r < rows; r += 3) {
    ctx.fillText('C' + (r + 1), 4, r * ch + ch * 0.75);
  }
}

function mfccColor(t) {
  // viridis-inspired: dark blue → teal → yellow
  const stops = [
    [13, 8, 135],    // 0.0
    [72, 40, 120],   // 0.2
    [63, 100, 175],  // 0.4
    [33, 145, 140],  // 0.6
    [94, 201, 98],   // 0.8
    [253, 231, 37],  // 1.0
  ];
  const n = stops.length - 1;
  const i = Math.min(Math.floor(t * n), n - 1);
  const f = t * n - i;
  const a = stops[i], b = stops[i + 1];
  const r = Math.round(a[0] + (b[0]-a[0]) * f);
  const g = Math.round(a[1] + (b[1]-a[1]) * f);
  const bv = Math.round(a[2] + (b[2]-a[2]) * f);
  return `rgb(${r},${g},${bv})`;
}

// ── Reset ─────────────────────────────────────────────────────
function resetUI() {
  selectedFile = null; demoMode = false;
  document.getElementById('uploadPanel').style.display = 'block';
  document.getElementById('progressPanel').style.display = 'none';
  document.getElementById('resultsPanel').style.display = 'none';
  clearFile();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Helpers ───────────────────────────────────────────────────
function formatSize(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
function scoreGradient(v) {
  if (v > 0.65) return '#ff3d5a';
  if (v > 0.40) return '#ffb800';
  return '#00e676';
}
function riskColor(level, alpha) {
  const map = { CRITICAL: `rgba(255,61,90,${alpha})`, HIGH: `rgba(255,61,90,${alpha})`, MODERATE: `rgba(255,184,0,${alpha})`, LOW: `rgba(0,230,118,${alpha})`, MINIMAL: `rgba(0,230,118,${alpha})` };
  return map[level] || `rgba(255,255,255,${alpha})`;
}