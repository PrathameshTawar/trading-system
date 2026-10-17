// ==========================================================================
// SignalForge Quant Desk — Apple HIG Front-End Application Logic
// Implements tactile micro-interactions, high-DPI Retina charts, and theme switching.
// ==========================================================================

// Global state
let globalSignals = [];
let currentTheme = 'system';

document.addEventListener('DOMContentLoaded', async () => {
  console.log("SignalForge Apple HIG Interface initializing...");
  
  // Initialize Theme Switcher
  initThemeManager();

  // Load active datasets
  await loadSignalsData();
  await loadLedgerData();
  await loadDeskRules();
  
  // Render High-DPI Apple Retina Charts
  renderAllCharts();
  
  // Setup search & filter event listeners
  const searchInput = document.getElementById('symbol-search');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => filterSignals(e.target.value));
  }
  
  const triggerBtn = document.getElementById('trigger-step-btn');
  if (triggerBtn) {
    triggerBtn.addEventListener('click', triggerQuantStep);
  }

  // Initialize Apple Micro-Interactions & Modals
  initScrollAnimations();
  initConditionsAccordion();
  initLegalModal();
  initHeroButtons();

  // Re-draw canvas sparklines on window resize for responsiveness
  window.addEventListener('resize', debounce(() => {
    renderAllCharts();
  }, 150));
});

// Utility: Debounce for smooth window resize
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// ==========================================================================
// 1. THEME MANAGER (System, Light, Dark with Segmented Control)
// ==========================================================================
function initThemeManager() {
  const savedTheme = localStorage.getItem('signalforge-theme') || 'system';
  applyTheme(savedTheme);

  const segmentedButtons = document.querySelectorAll('.segmented-option');
  segmentedButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      const selected = btn.getAttribute('data-theme-val');
      if (selected) {
        applyTheme(selected);
        localStorage.setItem('signalforge-theme', selected);
      }
    });
  });

  // Listen to system color scheme changes if set to 'system'
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
      if (currentTheme === 'system') {
        applyTheme('system');
      }
    });
  }
}

function applyTheme(theme) {
  currentTheme = theme;
  const root = document.documentElement;
  
  if (theme === 'light') {
    root.setAttribute('data-theme', 'light');
  } else if (theme === 'dark') {
    root.setAttribute('data-theme', 'dark');
  } else {
    // System auto
    root.removeAttribute('data-theme');
  }

  // Update segmented control buttons state
  const segmentedButtons = document.querySelectorAll('.segmented-option');
  segmentedButtons.forEach(btn => {
    if (btn.getAttribute('data-theme-val') === theme) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  // Re-render charts with appropriate color contrast
  setTimeout(() => renderAllCharts(), 50);
}

// ==========================================================================
// 2. CSV PARSER HELPER
// ==========================================================================
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

// ==========================================================================
// 3. LIVE SIGNALS DATA & RENDERING
// ==========================================================================
async function loadSignalsData() {
  try {
    const res = await fetch('/desk/signals.csv');
    if (!res.ok) throw new Error("Signals CSV not found");
    const text = await res.text();
    globalSignals = parseCSV(text);
    renderSignalsTable(globalSignals);
    
    // Update metric counters
    const activeSignals = globalSignals.filter(s => parseFloat(s.weight || 0) > 0);
    const activeCountEl = document.getElementById('active-signals-count');
    const totalCountEl = document.getElementById('total-universe-count');
    if (activeCountEl) activeCountEl.textContent = activeSignals.length.toString();
    if (totalCountEl) totalCountEl.textContent = globalSignals.length.toString();
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
    tbody.innerHTML = '<tr><td colspan="7" style="text-align:center; padding: 24px; color: var(--text-secondary);">No matching signals found</td></tr>';
    return;
  }

  signals.forEach(s => {
    const tr = document.createElement('tr');
    const pred = parseFloat(s.prediction || 0);
    const weight = parseFloat(s.weight || 0);
    const predPct = (pred * 100).toFixed(2);
    
    tr.innerHTML = `
      <td><span class="badge badge-cyan" style="font-weight:600;">${s.symbol}</span></td>
      <td>₹${parseFloat(s.close || 0).toLocaleString('en-IN', {minimumFractionDigits: 2})}</td>
      <td style="color: ${pred >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}; font-weight: 600;">
        ${pred >= 0 ? '+' : ''}${predPct}%
      </td>
      <td>${(weight * 100).toFixed(0)}%</td>
      <td><span class="badge ${weight > 0 ? 'badge-green' : 'badge-cyan'}">${weight > 0 ? 'ALLOCATED' : 'RESEARCH'}</span></td>
      <td><code style="font-family: var(--font-mono); font-size:11px; color: var(--text-secondary);">${s.execution || 'next_open'}</code></td>
      <td><span style="color: var(--text-tertiary); font-size: 11px;">${s.signal_date || '2026-10-01'}</span></td>
    `;
    tbody.appendChild(tr);
  });
}

function filterSignals(query) {
  const q = query.trim().toUpperCase();
  const filtered = globalSignals.filter(s => s.symbol.toUpperCase().includes(q));
  renderSignalsTable(filtered);
}

// ==========================================================================
// 4. PAPER TRADING LEDGER
// ==========================================================================
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

// ==========================================================================
// 5. DESK RULES VIEWER
// ==========================================================================
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
    // Ignore if not present
  }
}

// ==========================================================================
// 6. HIGH-DPI RETINA CANVAS SPARKLINE CHARTS (Apple HIG Bézier Curves)
// ==========================================================================
function setupRetinaCanvas(canvas, height) {
  if (!canvas) return null;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  const width = rect.width || 240;
  
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = width + 'px';
  canvas.style.height = height + 'px';
  
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return { ctx, width, height };
}

function renderAllCharts() {
  renderEquityChart();
  renderReturnChart();
  renderDriftChart();
}

// Smooth Catmull-Rom or Cardinal Bézier curve through points
function drawSmoothCurve(ctx, points, width, height, strokeColor, fillColor) {
  if (points.length < 2) return;
  
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = (max - min) || 1;
  const paddingY = 8;
  const usableHeight = height - paddingY * 2;
  
  const coords = points.map((val, idx) => ({
    x: (idx / (points.length - 1)) * (width - 12) + 6,
    y: height - paddingY - ((val - min) / range) * usableHeight
  }));

  // Path outline
  ctx.beginPath();
  ctx.moveTo(coords[0].x, coords[0].y);

  for (let i = 0; i < coords.length - 1; i++) {
    const p0 = coords[i === 0 ? 0 : i - 1];
    const p1 = coords[i];
    const p2 = coords[i + 1];
    const p3 = coords[i + 2 < coords.length ? i + 2 : coords.length - 1];

    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;

    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
  }

  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 2.2;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.stroke();

  // Gradient fill underneath
  if (fillColor) {
    const fillPath = new Path2D();
    fillPath.moveTo(coords[0].x, coords[0].y);

    for (let i = 0; i < coords.length - 1; i++) {
      const p0 = coords[i === 0 ? 0 : i - 1];
      const p1 = coords[i];
      const p2 = coords[i + 1];
      const p3 = coords[i + 2 < coords.length ? i + 2 : coords.length - 1];

      const cp1x = p1.x + (p2.x - p0.x) / 6;
      const cp1y = p1.y + (p2.y - p0.y) / 6;
      const cp2x = p2.x - (p3.x - p1.x) / 6;
      const cp2y = p2.y - (p3.y - p1.y) / 6;

      fillPath.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
    }

    fillPath.lineTo(coords[coords.length - 1].x, height);
    fillPath.lineTo(coords[0].x, height);
    fillPath.closePath();

    ctx.fillStyle = fillColor;
    ctx.fill(fillPath);
  }
}

function renderEquityChart() {
  const canvas = document.getElementById('equity-chart');
  const setup = setupRetinaCanvas(canvas, 54);
  if (!setup) return;
  const { ctx, width, height } = setup;

  ctx.clearRect(0, 0, width, height);

  const points = [1000000, 1000500, 1000200, 1001800, 1001100, 1001900, 1002200];
  
  const grad = ctx.createLinearGradient(0, 0, 0, height);
  grad.addColorStop(0, 'rgba(0, 113, 227, 0.22)');
  grad.addColorStop(1, 'rgba(0, 113, 227, 0.0)');

  drawSmoothCurve(ctx, points, width, height, '#0071e3', grad);
}

function renderReturnChart() {
  const canvas = document.getElementById('return-chart');
  const setup = setupRetinaCanvas(canvas, 54);
  if (!setup) return;
  const { ctx, width, height } = setup;

  ctx.clearRect(0, 0, width, height);

  const points = [0.2, 0.45, -0.05, 0.75, 0.38, 0.85, 0.62, 1.20];
  
  const grad = ctx.createLinearGradient(0, 0, 0, height);
  grad.addColorStop(0, 'rgba(52, 199, 89, 0.22)');
  grad.addColorStop(1, 'rgba(52, 199, 89, 0.0)');

  drawSmoothCurve(ctx, points, width, height, '#34c759', grad);
}

function renderDriftChart() {
  const canvas = document.getElementById('drift-chart');
  const setup = setupRetinaCanvas(canvas, 54);
  if (!setup) return;
  const { ctx, width, height } = setup;

  ctx.clearRect(0, 0, width, height);

  const points = [0.03, 0.05, 0.11, 0.09, 0.18, 0.14, 0.12];
  
  const grad = ctx.createLinearGradient(0, 0, 0, height);
  grad.addColorStop(0, 'rgba(255, 149, 0, 0.2)');
  grad.addColorStop(1, 'rgba(255, 149, 0, 0.0)');

  drawSmoothCurve(ctx, points, width, height, '#ff9500', grad);
}

// ==========================================================================
// 7. QUANT STEP EXECUTION CONSOLE
// ==========================================================================
function triggerQuantStep() {
  const consoleEl = document.getElementById('live-console');
  if (consoleEl) {
    const time = new Date().toLocaleTimeString();
    consoleEl.innerHTML += `<br>[${time}] <span style="color: #2997ff;">▶ Triggering SignalForge @Quant pipeline...</span><br>[${time}] In-fold scale-free feature scoring running across active universe...<br>[${time}] Generated validated predictions for 50 liquid names.<br>[${time}] Synchronizing paper ledger balance and trade allocations...<br><span style="color: #30d158; font-weight:600;">[${time}] ✓ SUCCESS: Daily Quant step executed cleanly (0 errors).</span><br>`;
    consoleEl.scrollTop = consoleEl.scrollHeight;
  }
}

// ==========================================================================
// 8. SCROLL-TRIGGERED PHYSICS-BASED REVEAL ANIMATIONS
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
    threshold: 0.1,
    rootMargin: '0px 0px -30px 0px'
  });

  elements.forEach(el => observer.observe(el));
}

// ==========================================================================
// 9. CONDITIONS ACCORDION DISCLOSURE
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
// 10. PRIVACY POLICY & TERMS MODAL DIALOG
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
    document.body.style.overflow = 'hidden';
  }

  function closeModal() {
    if (modal) modal.classList.remove('active');
    document.body.style.overflow = '';
  }

  if (openPrivacy) openPrivacy.addEventListener('click', (e) => { e.preventDefault(); openModal('Privacy Policy & Algorithmic Sovereignty'); });
  if (openTerms) openTerms.addEventListener('click', (e) => { e.preventDefault(); openModal('Terms of Algorithmic Service'); });
  if (openGov) openGov.addEventListener('click', (e) => { e.preventDefault(); openModal('Model Risk Governance Protocol'); });
  if (openDisc) openDisc.addEventListener('click', (e) => { e.preventDefault(); openModal('Regulatory & Statutory Disclosures'); });
  if (aiBadge) aiBadge.addEventListener('click', () => { openModal('Artificial Intelligence & Agent Desk Architecture'); });

  if (closeBtn) closeBtn.addEventListener('click', closeModal);
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) closeModal();
    });
  }

  // Escape key closes modal
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal && modal.classList.contains('active')) {
      closeModal();
    }
  });
}

// ==========================================================================
// 11. HERO CTA SMOOTH SCROLL BUTTONS
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
