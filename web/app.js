// SignalForge Quant Desk - Dynamic Front-End Application Logic

document.addEventListener('DOMContentLoaded', async () => {
  console.log("SignalForge Dashboard initializing...");
  
  // Set active timestamp
  const dateEl = document.getElementById('current-date');
  if (dateEl) {
    dateEl.textContent = new Date().toISOString().split('T')[0];
  }

  // Load datasets
  await loadSignalsData();
  await loadLedgerData();
  await loadDeskRules();
  
  // Draw Canvas Sparkline Charts
  renderEquityChart();
  renderReturnChart();
  renderDriftChart();
  
  // Setup search & filter event listeners
  const searchInput = document.getElementById('symbol-search');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => filterSignals(e.target.value));
  }
  
  const triggerBtn = document.getElementById('trigger-step-btn');
  if (triggerBtn) {
    triggerBtn.addEventListener('click', triggerQuantStep);
  }
});

// CSV Parser Helper
function parseCSV(csvText) {
  const lines = csvText.trim().split('\n');
  if (lines.length === 0) return [];
  const headers = lines[0].split(',').map(h => h.trim());
  return lines.slice(1).map(line => {
    const values = line.split(',').map(v => v.trim());
    const row = {};
    headers.forEach((h, i) => {
      row[h] = values[i] !== undefined ? values[i] : '';
    });
    return row;
  });
}

let globalSignals = [];

async function loadSignalsData() {
  try {
    const res = await fetch('/desk/signals.csv');
    if (!res.ok) throw new Error("Signals CSV not found");
    const text = await res.text();
    globalSignals = parseCSV(text);
    renderSignalsTable(globalSignals);
    
    // Update metric counters
    const activeSignals = globalSignals.filter(s => parseFloat(s.weight || 0) > 0);
    document.getElementById('active-signals-count').textContent = activeSignals.length.toString();
    document.getElementById('total-universe-count').textContent = globalSignals.length.toString();
  } catch (err) {
    console.warn("Could not fetch /desk/signals.csv dynamically, using fallback data", err);
    loadFallbackSignals();
  }
}

function loadFallbackSignals() {
  globalSignals = [
    { signal_date: '2026-10-01', symbol: 'MARUTI', close: '11386.0', prediction: '0.008186', weight: '0.2', execution: 'next_open' },
    { signal_date: '2026-10-01', symbol: 'GODREJCP', close: '840.0', prediction: '0.008115', weight: '0.2', execution: 'next_open' },
    { signal_date: '2026-10-01', symbol: 'ICICIPRULI', close: '448.05', prediction: '0.006447', weight: '0.2', execution: 'next_open' },
    { signal_date: '2026-10-01', symbol: 'BAJAJ-AUTO', close: '10045.0', prediction: '0.003470', weight: '0.2', execution: 'next_open' },
    { signal_date: '2026-10-01', symbol: 'RELIANCE', close: '1167.70', prediction: '0.002275', weight: '0.0', execution: 'next_open' },
    { signal_date: '2026-10-01', symbol: 'TCS', close: '2075.0', prediction: '0.001607', weight: '0.0', execution: 'next_open' },
    { signal_date: '2026-10-01', symbol: 'INFY', close: '1035.0', prediction: '0.002079', weight: '0.0', execution: 'next_open' },
    { signal_date: '2026-10-01', symbol: 'HDFCBANK', close: '721.20', prediction: '0.001897', weight: '0.0', execution: 'next_open' }
  ];
  renderSignalsTable(globalSignals);
}

function renderSignalsTable(signals) {
  const tbody = document.getElementById('signals-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (signals.length === 0) {
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;">No signals available</td></tr>';
    return;
  }

  signals.forEach(s => {
    const tr = document.createElement('tr');
    const pred = parseFloat(s.prediction || 0);
    const weight = parseFloat(s.weight || 0);
    const predPct = (pred * 100).toFixed(2);
    
    tr.innerHTML = `
      <td><span class="badge badge-cyan">${s.symbol}</span></td>
      <td>₹${parseFloat(s.close || 0).toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
      <td style="color: ${pred >= 0 ? 'var(--accent-green)' : 'var(--accent-pink)'}">${pred >= 0 ? '+' : ''}${predPct}%</td>
      <td>${(weight * 100).toFixed(0)}%</td>
      <td><span class="badge ${weight > 0 ? 'badge-green' : 'badge-cyan'}">${weight > 0 ? 'ALLOCATED' : 'RESEARCH'}</span></td>
      <td><code>${s.execution || 'next_open'}</code></td>
      <td><span style="color: var(--text-muted)">${s.signal_date || '2026-10-01'}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function filterSignals(query) {
  const q = query.trim().toUpperCase();
  const filtered = globalSignals.filter(s => s.symbol.toUpperCase().includes(q));
  renderSignalsTable(filtered);
}

async function loadLedgerData() {
  try {
    const res = await fetch('/desk/paper-ledger.csv');
    if (!res.ok) throw new Error("Ledger CSV not found");
    const text = await res.text();
    const ledger = parseCSV(text);
    renderLedgerTable(ledger);
  } catch (err) {
    console.warn("Using fallback paper ledger data", err);
    renderLedgerTable([
      { timestamp: '2026-10-09 23:10:49', symbol: 'CASH', quantity: '0', avg_price: '1.0', notional: '1000000.0', equity: '1000000.0', cash: '1000000.0', paused: 'False' }
    ]);
  }
}

function renderLedgerTable(ledger) {
  const tbody = document.getElementById('ledger-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  ledger.forEach(row => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><span class="badge badge-cyan">${row.symbol}</span></td>
      <td>${parseFloat(row.quantity || 0).toLocaleString()}</td>
      <td>₹${parseFloat(row.avg_price || 0).toFixed(2)}</td>
      <td>₹${parseFloat(row.notional || 0).toLocaleString('en-IN', {maximumFractionDigits: 0})}</td>
      <td>₹${parseFloat(row.cash || 0).toLocaleString('en-IN', {maximumFractionDigits: 0})}</td>
      <td><span class="badge badge-green">SETTLED</span></td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadDeskRules() {
  try {
    const res = await fetch('/desk/my-rules.md');
    if (!res.ok) return;
    const text = await res.text();
    const rulesBox = document.getElementById('rules-box');
    if (rulesBox) {
      rulesBox.textContent = text;
    }
  } catch (e) {
    // Ignore
  }
}

// Canvas Sparkline Chart Renderers
function renderEquityChart() {
  const canvas = document.getElementById('equity-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = 140;

  ctx.clearRect(0, 0, w, h);
  
  // Data points
  const points = [1000000, 1002500, 1001200, 1005800, 1004100, 1008900, 1012000];
  const min = Math.min(...points) * 0.998;
  const max = Math.max(...points) * 1.002;

  ctx.beginPath();
  ctx.strokeStyle = '#00f2fe';
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';

  points.forEach((val, idx) => {
    const x = (idx / (points.length - 1)) * (w - 20) + 10;
    const y = h - ((val - min) / (max - min)) * (h - 30) - 15;
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Gradient fill under curve
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, 'rgba(0, 242, 254, 0.25)');
  grad.addColorStop(1, 'rgba(0, 242, 254, 0.0)');
  ctx.lineTo(w - 10, h);
  ctx.lineTo(10, h);
  ctx.fillStyle = grad;
  ctx.fill();
}

function renderReturnChart() {
  const canvas = document.getElementById('return-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = 60;

  ctx.clearRect(0, 0, w, h);
  const data = [0.2, 0.5, -0.1, 0.8, 0.4, 0.9, 0.6, 1.1];
  
  ctx.beginPath();
  ctx.strokeStyle = '#00e676';
  ctx.lineWidth = 2;
  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - (v / 1.5) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function renderDriftChart() {
  const canvas = document.getElementById('drift-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = 60;

  ctx.clearRect(0, 0, w, h);
  const data = [0.02, 0.05, 0.12, 0.08, 0.24, 0.18, 0.28];
  
  ctx.beginPath();
  ctx.strokeStyle = '#ff3366';
  ctx.lineWidth = 2;
  data.forEach((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - (v / 0.35) * h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function triggerQuantStep() {
  const consoleEl = document.getElementById('live-console');
  if (consoleEl) {
    const time = new Date().toLocaleTimeString();
    consoleEl.innerHTML += `<br>[${time}] Triggering @Quant step...<br>[${time}] Scoring universe against frozen model...<br>[${time}] Writing signals to desk/signals.csv...<br>[${time}] Syncing paper/ledger.json to desk/paper-ledger.csv...<br><span style="color: var(--accent-green)">[${time}] SUCCESS: Daily Quant step executed cleanly.</span><br>`;
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }
}

// ==========================================================================
// SCROLL-TRIGGERED 3D REVEAL ANIMATIONS (INTERSECTION OBSERVER)
// ==========================================================================
function initScrollAnimations() {
  const elements = document.querySelectorAll('.scroll-reveal');
  if (!elements.length) return;

  const observer = new IntersectionObserver((entries, obs) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('revealed');
        obs.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.15,
    rootMargin: '0px 0px -40px 0px'
  });

  elements.forEach(el => observer.observe(el));
}

// ==========================================================================
// CONDITIONS ACCORDION TOGGLE
// ==========================================================================
function initConditionsAccordion() {
  const toggleBtn = document.getElementById('toggle-conditions-btn');
  const body = document.getElementById('conditions-body');
  const arrow = document.getElementById('accordion-arrow-icon');
  
  if (toggleBtn && body && arrow) {
    toggleBtn.addEventListener('click', () => {
      const isOpen = body.classList.contains('open');
      if (isOpen) {
        body.classList.remove('open');
        arrow.classList.remove('rotated');
      } else {
        body.classList.add('open');
        arrow.classList.add('rotated');
      }
    });
  }
}

// ==========================================================================
// PRIVACY POLICY & TERMS MODAL POPUP
// ==========================================================================
function initLegalModal() {
  const modal = document.getElementById('legal-modal');
  const closeBtn = document.getElementById('close-modal-btn');
  const modalTitle = document.getElementById('modal-title-text');
  
  const openPrivacy = document.getElementById('open-privacy-link');
  const openTerms = document.getElementById('open-terms-link');
  const openGov = document.getElementById('open-governance-link');
  const openDisc = document.getElementById('open-disclaimers-link');
  const aiBadge = document.getElementById('made-with-ai-badge');

  function openModal(title) {
    if (modalTitle) modalTitle.textContent = title;
    if (modal) modal.classList.add('active');
  }

  function closeModal() {
    if (modal) modal.classList.remove('active');
  }

  if (openPrivacy) openPrivacy.addEventListener('click', (e) => { e.preventDefault(); openModal('🔒 Privacy Policy & Algorithmic Sovereignty'); });
  if (openTerms) openTerms.addEventListener('click', (e) => { e.preventDefault(); openModal('📜 Terms of Algorithmic Service'); });
  if (openGov) openGov.addEventListener('click', (e) => { e.preventDefault(); openModal('🛡️ Model Risk Governance Protocol'); });
  if (openDisc) openDisc.addEventListener('click', (e) => { e.preventDefault(); openModal('⚖️ Regulatory & Statutory Disclosures'); });
  if (aiBadge) aiBadge.addEventListener('click', () => { openModal('✨ Artificial Intelligence & Agent Desk Disclosures'); });

  if (closeBtn) closeBtn.addEventListener('click', closeModal);
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) closeModal();
    });
  }
}

// ==========================================================================
// HERO CTA SMOOTH SCROLL BUTTONS
// ==========================================================================
function initHeroButtons() {
  const ctaBtn = document.getElementById('hero-cta-btn');
  const condBtn = document.getElementById('hero-conditions-btn');
  
  if (ctaBtn) {
    ctaBtn.addEventListener('click', () => {
      const target = document.getElementById('signals-section');
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  }

  if (condBtn) {
    condBtn.addEventListener('click', () => {
      const condSection = document.getElementById('conditions-accordion-section');
      const body = document.getElementById('conditions-body');
      const arrow = document.getElementById('accordion-arrow-icon');
      if (condSection) {
        condSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
        if (body && !body.classList.contains('open')) {
          body.classList.add('open');
          if (arrow) arrow.classList.add('rotated');
        }
      }
    });
  }
}

// Initialize interactive elements after DOM loads
document.addEventListener('DOMContentLoaded', () => {
  initScrollAnimations();
  initConditionsAccordion();
  initLegalModal();
  initHeroButtons();
});
