// ---------------------------------------------------------------------------
// CONFIG - point this at your deployed backend URL before publishing.
// Locally this matches `uvicorn main:app --port 8000`.
// ---------------------------------------------------------------------------
const API_BASE = "https://treatment-advisor.onrender.com";

const GAUGE_ARC_LENGTH = 267; // half-circumference of the gauge path, in SVG units
const RISK_COLORS = { Low: "#4F7A5E", Moderate: "#B8863C", High: "#A8432B" };

let currentCondition = "diabetes";
let currentFeatures = {};
let debounceTimer = null;
let comparisonVisible = false;
let comparisonCache = {};

const slidersEl = document.getElementById("sliders");
const factorsEl = document.getElementById("factors");
const recommendationsEl = document.getElementById("recommendations");
const riskPercentEl = document.getElementById("risk-percent");
const riskTierEl = document.getElementById("risk-tier");
const metricsLineEl = document.getElementById("metrics-line");
const disclaimerEl = document.getElementById("disclaimer");
const gaugeFillEl = document.getElementById("gauge-fill");
const uncertaintyBannerEl = document.getElementById("uncertainty-banner");
const comparisonTableEl = document.getElementById("comparison-table");
const compareToggleEl = document.getElementById("compare-toggle");

compareToggleEl.addEventListener("click", async () => {
  comparisonVisible = !comparisonVisible;
  if (comparisonVisible) {
    await renderComparisonTable(currentCondition);
    comparisonTableEl.hidden = false;
    compareToggleEl.textContent = "hide model comparison";
  } else {
    comparisonTableEl.hidden = true;
    compareToggleEl.textContent = "view model comparison";
  }
});

async function renderComparisonTable(condition) {
  if (!comparisonCache[condition]) {
    const res = await fetch(`${API_BASE}/api/model-comparison/${condition}`);
    comparisonCache[condition] = await res.json();
  }
  const data = comparisonCache[condition];
  const modelLabels = {
    random_forest: "Random Forest",
    xgboost: "XGBoost",
    logistic_regression: "Logistic Regression",
  };

  let rows = "";
  Object.entries(data.comparison).forEach(([name, m]) => {
    const isWinner = name === data.production_model;
    rows += `
      <tr class="${isWinner ? "winner" : ""}">
        <td>${modelLabels[name] || name}${isWinner ? '<span class="winner-tag">selected</span>' : ""}</td>
        <td>${(m.accuracy * 100).toFixed(1)}%</td>
        <td>${(m.precision * 100).toFixed(1)}%</td>
        <td>${(m.recall * 100).toFixed(1)}%</td>
        <td>${(m.f1 * 100).toFixed(1)}%</td>
        <td>${m.roc_auc.toFixed(3)}</td>
      </tr>`;
  });

  comparisonTableEl.innerHTML = `
    <table>
      <thead><tr><th>Model (5-fold CV)</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="comparison-note">${data.selection_rule}</p>
  `;
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => {
      t.classList.remove("active");
      t.setAttribute("aria-selected", "false");
    });
    tab.classList.add("active");
    tab.setAttribute("aria-selected", "true");
    currentCondition = tab.dataset.condition;
    comparisonVisible = false;
    comparisonTableEl.hidden = true;
    compareToggleEl.textContent = "view model comparison";
    loadCondition(currentCondition);
  });
});

async function loadCondition(condition) {
  const res = await fetch(`${API_BASE}/api/meta/${condition}`);
  const meta = await res.json();
  buildSliders(meta);
  runPrediction();
}

function buildSliders(meta) {
  slidersEl.innerHTML = "";
  currentFeatures = {};
  meta.feature_names.forEach((name) => {
    const cfg = meta.feature_meta[name];
    currentFeatures[name] = cfg.default;

    const row = document.createElement("div");
    row.className = "slider-row";
    row.innerHTML = `
      <div class="slider-label">
        <span>${cfg.label}</span>
        <span data-out="${name}">${cfg.default}</span>
      </div>
      <input type="range" min="${cfg.min}" max="${cfg.max}" step="${cfg.step}" value="${cfg.default}" data-feature="${name}">
    `;
    slidersEl.appendChild(row);
  });

  slidersEl.querySelectorAll("input[type=range]").forEach((input) => {
    input.addEventListener("input", (e) => {
      const name = e.target.dataset.feature;
      const value = parseFloat(e.target.value);
      currentFeatures[name] = value;
      slidersEl.querySelector(`[data-out="${name}"]`).textContent = value;
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(runPrediction, 200);
    });
  });
}

async function runPrediction() {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ condition: currentCondition, features: currentFeatures }),
  });
  if (!res.ok) return;
  const data = await res.json();
  renderResult(data);
}

function renderResult(data) {
  const pct = Math.round(data.risk_probability * 100);
  const color = RISK_COLORS[data.risk_tier] || "#4F7A5E";

  riskPercentEl.textContent = `${pct}%`;
  riskTierEl.textContent = `${data.risk_tier} risk`;
  gaugeFillEl.style.stroke = color;
  const offset = GAUGE_ARC_LENGTH * (1 - data.risk_probability);
  gaugeFillEl.style.strokeDasharray = `${GAUGE_ARC_LENGTH}`;
  gaugeFillEl.style.strokeDashoffset = `${offset}`;

  const m = data.model_metrics;
  metricsLineEl.textContent = `${data.production_model.replace("_", " ")} · accuracy ${(m.accuracy * 100).toFixed(0)}% · ROC-AUC ${m.roc_auc.toFixed(2)}`;

  if (data.uncertainty && data.uncertainty.is_uncertain) {
    uncertaintyBannerEl.textContent = data.uncertainty.message;
    uncertaintyBannerEl.hidden = false;
  } else {
    uncertaintyBannerEl.hidden = true;
  }

  const maxAbs = Math.max(...data.top_factors.map((f) => Math.abs(f.contribution)), 0.001);
  factorsEl.innerHTML = "";
  data.top_factors.forEach((f) => {
    const widthPct = (Math.abs(f.contribution) / maxAbs) * 50;
    const isPositive = f.contribution >= 0;
    const barColor = isPositive ? color : "#9A9481";
    const row = document.createElement("div");
    row.className = "factor-row";
    row.innerHTML = `
      <span>${f.label}</span>
      <div class="factor-track">
        <div class="factor-fill" style="
          background:${barColor};
          left:${isPositive ? 50 : 50 - widthPct}%;
          width:${widthPct}%;
        "></div>
      </div>
      <span class="factor-value">${f.contribution > 0 ? "+" : ""}${f.contribution.toFixed(3)}</span>
    `;
    factorsEl.appendChild(row);
  });

  recommendationsEl.innerHTML = "";
  data.recommendations.forEach((r) => {
    const li = document.createElement("li");
    li.textContent = r;
    recommendationsEl.appendChild(li);
  });

  disclaimerEl.textContent = data.disclaimer;
}

loadCondition(currentCondition);
