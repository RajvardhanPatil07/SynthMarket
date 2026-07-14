const form = document.querySelector("#runForm");
const runButton = document.querySelector("#runButton");
const healthDot = document.querySelector("#healthDot");
const statusBadge = document.querySelector("#statusBadge");
const runMessage = document.querySelector("#runMessage");
const stageText = document.querySelector("#stageText");
const elapsedText = document.querySelector("#elapsedText");
const progressBar = document.querySelector("#progressBar");
const metricStatus = document.querySelector("#metricStatus");
const metricKs = document.querySelector("#metricKs");
const metricVol = document.querySelector("#metricVol");
const metricMemo = document.querySelector("#metricMemo");
const downloadActions = document.querySelector("#downloadActions");
const warningList = document.querySelector("#warningList");
const outputHint = document.querySelector("#outputHint");
const plotGrid = document.querySelector("#plotGrid");
const pathChart = document.querySelector("#pathChart");
const returnChart = document.querySelector("#returnChart");
const equityChart = document.querySelector("#equityChart");
const strategyReturnChart = document.querySelector("#strategyReturnChart");
const drawdownChart = document.querySelector("#drawdownChart");
const pathHint = document.querySelector("#pathHint");
const btRobustness = document.querySelector("#btRobustness");
const btMedian = document.querySelector("#btMedian");
const btProfitable = document.querySelector("#btProfitable");
const btDrawdown = document.querySelector("#btDrawdown");
const worstPathRows = document.querySelector("#worstPathRows");
const strategyTemplate = document.querySelector("#strategyTemplate");
const strategyName = document.querySelector("#strategyName");
const strategyParamGrid = document.querySelector("#strategyParamGrid");
const templatePalette = document.querySelector("#templatePalette");
const builderDropZone = document.querySelector("#builderDropZone");
const builderTemplateName = document.querySelector("#builderTemplateName");
const builderTemplateDescription = document.querySelector("#builderTemplateDescription");
const advancedCode = document.querySelector("#advancedCode");
const saveStrategyButton = document.querySelector("#saveStrategyButton");
const strategySaveStatus = document.querySelector("#strategySaveStatus");
const runRows = document.querySelector("#runRows");
const strategyLibrary = document.querySelector("#strategyLibrary");
const strategyLibraryCount = document.querySelector("#strategyLibraryCount");
const compareSelector = document.querySelector("#compareSelector");
const compareButton = document.querySelector("#compareButton");
const compareStatus = document.querySelector("#compareStatus");
const compareRows = document.querySelector("#compareRows");
const themeToggle = document.querySelector("#themeToggle");
const themeIcon = document.querySelector("#themeIcon");
const activeStrategyLabel = document.querySelector("#activeStrategyLabel");
const savedBacktestBar = document.querySelector("#savedBacktestBar");
const savedBacktestLabel = document.querySelector("#savedBacktestLabel");
const savedBackButton = document.querySelector("#savedBackButton");
const savedPrevRunButton = document.querySelector("#savedPrevRunButton");
const savedNextRunButton = document.querySelector("#savedNextRunButton");

let activeJobId = null;
let pollTimer = null;
let lastJob = null;
let strategyTemplates = [];
let savedRuns = [];
let selectedSavedRunId = null;
let savedBacktestReturnTab = "projectsTab";

const storedTheme = localStorage.getItem("synthmarket-theme");
const preferredDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
applyTheme(storedTheme || (preferredDark ? "dark" : "light"));

const presets = {
  fast: {
    ticker: "SPY,QQQ",
    generation_mode: "correlated_multi_asset",
    period: "1y",
    epochs: 1,
    window_size: 10,
    n_paths: 2,
    length: 30,
    hidden_dim: 8,
    batch_size: 4,
    n_critic: 1,
    noise_dim: 4,
    num_layers: 1,
    sma_short: 3,
    sma_long: 8,
  },
  balanced: {
    ticker: "SPY,QQQ",
    generation_mode: "correlated_multi_asset",
    period: "5y",
    epochs: 10,
    window_size: 60,
    n_paths: 100,
    length: 126,
    hidden_dim: 64,
    batch_size: 32,
    n_critic: 3,
    noise_dim: 16,
    num_layers: 1,
    sma_short: 20,
    sma_long: 50,
  },
  quality: {
    ticker: "SPY,QQQ",
    generation_mode: "correlated_multi_asset",
    period: "10y",
    epochs: 100,
    window_size: 252,
    n_paths: 1000,
    length: 252,
    hidden_dim: 128,
    batch_size: 64,
    n_critic: 5,
    noise_dim: 16,
    num_layers: 2,
    sma_short: 20,
    sma_long: 50,
  },
};

const plotLabels = {
  return_distribution: "Return Distribution",
  qq: "QQ Plot",
  return_acf: "Return ACF",
  squared_return_acf: "Squared Return ACF",
  rolling_volatility: "Rolling Volatility",
  path_preview: "Path Preview",
};

const tabButtons = [...document.querySelectorAll(".tab-button")];

tabButtons.forEach((button, index) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.tabTarget));
  button.addEventListener("keydown", (event) => {
    const keyOffsets = { ArrowRight: 1, ArrowLeft: -1 };
    let nextIndex = index;
    if (event.key in keyOffsets) {
      nextIndex = (index + keyOffsets[event.key] + tabButtons.length) % tabButtons.length;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = tabButtons.length - 1;
    } else {
      return;
    }
    event.preventDefault();
    const nextButton = tabButtons[nextIndex];
    setActiveTab(nextButton.dataset.tabTarget);
    nextButton.focus({ preventScroll: true });
  });
});

document.querySelectorAll(".preset-button[data-preset]").forEach((button) => {
  button.addEventListener("click", () => applyPreset(button.dataset.preset));
});

strategyTemplate.addEventListener("change", () => renderStrategyTemplate(strategyTemplate.value));
saveStrategyButton.addEventListener("click", saveCurrentStrategy);
compareButton.addEventListener("click", compareSelectedRuns);
savedBackButton.addEventListener("click", backToSavedRuns);
savedPrevRunButton.addEventListener("click", () => viewAdjacentSavedRun(-1));
savedNextRunButton.addEventListener("click", () => viewAdjacentSavedRun(1));
themeToggle.addEventListener("click", () => {
  const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(nextTheme);
  redrawCharts();
});

builderDropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  builderDropZone.classList.add("dragging");
});

builderDropZone.addEventListener("dragleave", () => {
  builderDropZone.classList.remove("dragging");
});

builderDropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  builderDropZone.classList.remove("dragging");
  const templateId = event.dataTransfer.getData("text/plain");
  if (templateId) {
    strategyTemplate.value = templateId;
    renderStrategyTemplate(templateId);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const params = Object.fromEntries(new FormData(form).entries());
  params.tickers = params.ticker;
  params.strategy_spec = buildStrategySpec();
  params.portfolio_spec = buildPortfolioSpec(params);
  selectedSavedRunId = null;
  hideSavedBacktestBar();
  setBusy(true);
  setMessage("Starting run…");
  resetOutputs();

  try {
    const response = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Unable to start run");
    }
    activeJobId = payload.job.job_id;
    renderJob(payload.job);
    startPolling();
  } catch (error) {
    setBusy(false);
    setStatus("failed");
    setMessage(error.message);
  }
});

async function bootstrap() {
  try {
    await loadStrategyTemplates();
    await loadWorkspace();
    const health = await fetch("/api/health");
    healthDot.classList.toggle("online", health.ok);
    const latest = await fetch("/api/latest");
    const payload = await latest.json();
    if (payload.job) {
      activeJobId = payload.job.job_id;
      renderJob(payload.job);
      if (["queued", "running"].includes(payload.job.status)) {
        setBusy(true);
        startPolling();
      }
    } else {
      drawAllEmpty();
    }
  } catch {
    healthDot.classList.remove("online");
  }
}

function setActiveTab(targetId) {
  tabButtons.forEach((button) => {
    const isActive = button.dataset.tabTarget === targetId;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", String(isActive));
    button.tabIndex = isActive ? 0 : -1;
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    const isActive = panel.id === targetId;
    panel.classList.toggle("active", isActive);
    panel.hidden = !isActive;
  });
  redrawCharts();
}

function applyPreset(name) {
  const preset = presets[name];
  if (!preset) {
    return;
  }
  Object.entries(preset).forEach(([key, value]) => {
    const control = form.elements.namedItem(key);
    if (control) {
      control.value = String(value);
    }
  });
  if (strategyTemplate.value === "sma_crossover") {
    setStrategyParam("short_window", preset.sma_short);
    setStrategyParam("long_window", preset.sma_long);
  }
}

function startPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(fetchActiveJob, 1200);
}

async function fetchActiveJob() {
  if (!activeJobId) {
    return;
  }
  const response = await fetch(`/api/jobs/${activeJobId}`);
  const payload = await response.json();
  if (!response.ok) {
    setBusy(false);
    setMessage(payload.error || "Unable to load job");
    clearInterval(pollTimer);
    return;
  }
  renderJob(payload.job);
  if (!["queued", "running"].includes(payload.job.status)) {
    setBusy(false);
    clearInterval(pollTimer);
    loadWorkspace();
  }
}

function renderJob(job) {
  lastJob = job;
  setStatus(job.status);
  setMessage(job.message || job.stage);
  stageText.textContent = labelize(job.stage);
  elapsedText.textContent = `${Number(job.elapsed_seconds || 0).toFixed(2)}s`;
  const progress = Math.round((job.progress || 0) * 100);
  progressBar.style.transform = `scaleX(${progress / 100})`;
  progressBar.setAttribute("aria-valuenow", String(progress));

  if (job.metrics && Object.keys(job.metrics).length) {
    renderMetrics(job.metrics);
  }
  if (job.preview && Object.keys(job.preview).length) {
    pathHint.textContent = `${job.params?.n_paths || "-"} path(s), ${job.params?.length || "-"} bars`;
    drawPathChart(job.preview);
    drawReturnChart(job.preview);
  }
  if (job.backtest && Object.keys(job.backtest).length) {
    renderBacktest(job.backtest);
  }
  if (job.files && Object.keys(job.files).length) {
    renderFiles(job);
  }
  renderWarnings(job.warnings || []);
}

function renderMetrics(metrics) {
  metricStatus.textContent = String(metrics.status || "-").toUpperCase();
  metricKs.textContent = fmt(metrics.ks_statistic);
  metricVol.textContent = fmt(metrics.rolling_volatility_ratio);
  metricMemo.textContent = fmt(metrics.memorization?.near_duplicate_rate);
}

function renderBacktest(backtest) {
  const aggregate = backtest.aggregate || {};
  btRobustness.textContent = pct(aggregate.robustness_score);
  btMedian.textContent = pct(aggregate.median_return);
  btProfitable.textContent = pct(aggregate.percent_profitable);
  btDrawdown.textContent = pct(aggregate.worst_drawdown);
  drawEquityChart(backtest);
  drawHistogram(strategyReturnChart, [{ label: "Return", data: backtest.return_histogram, color: alphaColor("accent", 0.58) }]);
  drawHistogram(drawdownChart, [{ label: "Drawdown", data: backtest.drawdown_histogram, color: alphaColor("danger", 0.52) }]);
  renderWorstPaths(backtest.worst_paths || []);
}

function viewSavedRun(runId, targetTab = "backtestTab", returnTab = null) {
  const run = savedRuns.find((item) => String(item.run_id) === String(runId));
  if (!run) {
    return;
  }
  const activeTab = document.querySelector(".tab-button.active")?.dataset.tabTarget || "projectsTab";
  savedBacktestReturnTab = returnTab || (activeTab === "backtestTab" ? savedBacktestReturnTab : activeTab);
  selectedSavedRunId = run.run_id;
  const job = savedRunToJob(run);
  renderJob(job);
  renderRunRows(savedRuns);
  renderCompareSelector(savedRuns);
  showSavedBacktestBar(run);
  setMessage(`Viewing saved backtest ${run.run_id}`);
  setStatus("succeeded");
  setBusy(false);
  clearInterval(pollTimer);
  setActiveTab(targetTab);
}

function showSavedBacktestBar(run) {
  const index = savedRuns.findIndex((item) => String(item.run_id) === String(run.run_id));
  savedBacktestBar.classList.remove("hidden");
  savedBacktestLabel.textContent = `${run.strategy_name || "Strategy"} | ${run.run_id}`;
  savedBackButton.textContent = savedBacktestReturnTab === "compareTab" ? "Back to compare" : "Back to saved runs";
  savedPrevRunButton.disabled = index <= 0;
  savedNextRunButton.disabled = index < 0 || index >= savedRuns.length - 1;
}

function hideSavedBacktestBar() {
  savedBacktestBar.classList.add("hidden");
}

function backToSavedRuns() {
  setActiveTab(savedBacktestReturnTab || "projectsTab");
  renderRunRows(savedRuns);
  renderCompareSelector(savedRuns);
}

function viewAdjacentSavedRun(direction) {
  const index = savedRuns.findIndex((item) => String(item.run_id) === String(selectedSavedRunId));
  const next = savedRuns[index + direction];
  if (next) {
    viewSavedRun(next.run_id, "backtestTab", savedBacktestReturnTab);
  }
}

function savedRunToJob(run) {
  return {
    job_id: run.run_id,
    status: "succeeded",
    stage: "saved_backtest",
    message: `Viewing saved backtest ${run.run_id}`,
    elapsed_seconds: 0,
    progress: 1,
    params: run.params_json || {},
    metrics: run.metrics_json || {},
    backtest: run.backtest_json || {},
    files: run.files_json || {},
    warnings: [],
  };
}

function renderWorstPaths(rows) {
  if (!rows.length) {
    worstPathRows.innerHTML = '<tr><td colspan="6">-</td></tr>';
    return;
  }
  worstPathRows.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${row.path_id}</td>
          <td>${pct(row.total_return)}</td>
          <td>${fmt(row.sharpe)}</td>
          <td>${pct(row.max_drawdown)}</td>
          <td>${fmt(row.trade_count)}</td>
          <td>${money(row.final_equity)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderFiles(job) {
  outputHint.textContent = `artifacts/web/${job.job_id}`;
  downloadActions.innerHTML = "";

  if (job.files.synthetic_csv) {
    downloadActions.appendChild(fileLink(job.job_id, "synthetic_csv", "Synthetic CSV"));
  }
  if (job.files.backtest_csv) {
    downloadActions.appendChild(fileLink(job.job_id, "backtest_csv", "Backtest CSV"));
  }
  if (job.files.strategy_json) {
    downloadActions.appendChild(fileLink(job.job_id, "strategy_json", "Strategy JSON"));
  }
  if (job.files.vectorbt_csv) {
    downloadActions.appendChild(exportLink(job.job_id, "vectorbt", "VectorBT CSV"));
  }
  if (job.files.backtrader_zip) {
    downloadActions.appendChild(exportLink(job.job_id, "backtrader", "Backtrader ZIP"));
  }
  if (job.files.checkpoint) {
    downloadActions.appendChild(fileLink(job.job_id, "checkpoint", "Checkpoint"));
  }

  plotGrid.innerHTML = "";
  Object.keys(plotLabels).forEach((key) => {
    if (!job.files[key]) {
      return;
    }
    const link = document.createElement("a");
    link.href = `/api/files/${job.job_id}/${key}`;
    link.target = "_blank";
    const img = document.createElement("img");
    img.src = `/api/files/${job.job_id}/${key}`;
    img.alt = plotLabels[key];
    const span = document.createElement("span");
    span.textContent = plotLabels[key];
    link.append(img, span);
    plotGrid.appendChild(link);
  });
}

function fileLink(jobId, key, label) {
  const link = document.createElement("a");
  link.href = `/api/files/${jobId}/${key}`;
  link.textContent = label;
  return link;
}

function exportLink(jobId, key, label) {
  const link = document.createElement("a");
  link.href = `/api/export/${jobId}/${key}`;
  link.textContent = label;
  return link;
}

function renderWarnings(warnings) {
  warningList.innerHTML = "";
  warnings.forEach((warning) => {
    const node = document.createElement("div");
    node.className = "warning";
    node.textContent = warning;
    warningList.appendChild(node);
  });
}

async function loadStrategyTemplates() {
  const response = await fetch("/api/strategies/templates");
  const payload = await response.json();
  strategyTemplates = payload.templates || [];
  strategyTemplate.innerHTML = strategyTemplates
    .map((template) => `<option value="${template.id}">${template.label}</option>`)
    .join("");
  strategyTemplate.value = "sma_crossover";
  renderTemplatePalette();
  renderStrategyTemplate(strategyTemplate.value || strategyTemplates[0]?.id || "sma_crossover");
}

function renderTemplatePalette() {
  templatePalette.innerHTML = strategyTemplates
    .map(
      (template) => `
        <button class="template-chip" type="button" draggable="true" data-template-id="${template.id}">
          ${template.label}
        </button>
      `,
    )
    .join("");
  templatePalette.querySelectorAll(".template-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      strategyTemplate.value = chip.dataset.templateId;
      renderStrategyTemplate(chip.dataset.templateId);
    });
    chip.addEventListener("dragstart", (event) => {
      event.dataTransfer.setData("text/plain", chip.dataset.templateId);
    });
  });
}

function renderStrategyTemplate(templateId) {
  const template = strategyTemplates.find((item) => item.id === templateId);
  if (!template) {
    return;
  }
  activeStrategyLabel.textContent = template.label;
  builderTemplateName.textContent = template.label;
  builderTemplateDescription.textContent = template.description;
  strategyParamGrid.innerHTML = "";
  Object.entries(template.parameters || {}).forEach(([name, spec]) => {
    const label = document.createElement("label");
    const span = document.createElement("span");
    span.textContent = spec.label || name;
    const input = document.createElement("input");
    input.name = `strategy_param_${name}`;
    input.dataset.paramName = name;
    input.type = "number";
    input.min = spec.min;
    input.max = spec.max;
    input.step = spec.type === "int" ? "1" : "0.1";
    input.value = spec.default;
    label.append(span, input);
    strategyParamGrid.appendChild(label);
  });
}

function buildStrategySpec() {
  const template = strategyTemplate.value || "sma_crossover";
  const parameters = {};
  strategyParamGrid.querySelectorAll("input[data-param-name]").forEach((input) => {
    parameters[input.dataset.paramName] = Number(input.value);
  });
  return {
    name: strategyName.value || "Strategy",
    template,
    parameters,
    long_only: true,
    description: "",
    advanced_code: advancedCode.value || "",
  };
}

function buildPortfolioSpec(params) {
  return {
    initial_cash: Number(params.initial_cash || 10000),
    fee_bps: Number(params.fee_bps || 1),
    weights: parseWeights(params.portfolio_weights),
    rebalance_frequency: params.rebalance_frequency || "daily",
  };
}

function parseWeights(value) {
  if (!value || !value.trim()) {
    return null;
  }
  const weights = {};
  value.split(",").forEach((part) => {
    const trimmed = part.trim();
    if (!trimmed) {
      return;
    }
    const [ticker, weight] = trimmed.includes(":") ? trimmed.split(":") : [null, null];
    if (ticker && weight) {
      weights[ticker.trim().toUpperCase()] = Number(weight);
    }
  });
  return Object.keys(weights).length ? weights : null;
}

function setStrategyParam(name, value) {
  const input = strategyParamGrid.querySelector(`input[data-param-name="${name}"]`);
  if (input && value !== undefined) {
    input.value = String(value);
  }
}

async function saveCurrentStrategy() {
  strategySaveStatus.textContent = "Saving…";
  try {
    const response = await fetch("/api/strategies", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ strategy: buildStrategySpec() }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Unable to save strategy");
    }
    strategySaveStatus.textContent = "Saved";
    await loadWorkspace();
  } catch (error) {
    strategySaveStatus.textContent = error.message;
  }
}

async function loadWorkspace() {
  const [runsResponse, strategiesResponse] = await Promise.all([fetch("/api/runs"), fetch("/api/strategies")]);
  const runsPayload = await runsResponse.json();
  const strategiesPayload = await strategiesResponse.json();
  savedRuns = runsPayload.runs || [];
  renderRunRows(savedRuns);
  renderStrategyLibrary(strategiesPayload.strategies || []);
  renderCompareSelector(savedRuns);
}

function renderRunRows(runs) {
  if (!runs.length) {
    runRows.innerHTML = '<tr><td colspan="6">No saved runs yet</td></tr>';
    return;
  }
  runRows.innerHTML = runs
    .map((run) => {
      const aggregate = run.backtest_json?.aggregate || {};
      const selected = String(run.run_id) === String(selectedSavedRunId);
      return `
        <tr class="${selected ? "selected-row" : ""}">
          <td>${run.run_id}</td>
          <td>${run.strategy_name || "-"}</td>
          <td>${labelize(run.strategy_template || "-")}</td>
          <td>${pct(aggregate.median_return)}</td>
          <td>${pct(aggregate.robustness_score)}</td>
          <td>
            <button class="inline-button" type="button" data-view-run-id="${run.run_id}">
              ${selected ? "Viewing" : "View"}
            </button>
          </td>
        </tr>
      `;
    })
    .join("");
  runRows.querySelectorAll("[data-view-run-id]").forEach((button) => {
    button.addEventListener("click", () => viewSavedRun(button.dataset.viewRunId, "backtestTab", "projectsTab"));
  });
}

function renderStrategyLibrary(strategies) {
  strategyLibraryCount.textContent = `${strategies.length} saved`;
  if (!strategies.length) {
    strategyLibrary.innerHTML = '<div class="empty-note">No saved strategies yet</div>';
    return;
  }
  strategyLibrary.innerHTML = strategies
    .map(
      (strategy) => `
        <article class="library-card">
          <strong>${strategy.name}</strong>
          <span>${labelize(strategy.template)}</span>
        </article>
      `,
    )
    .join("");
}

function renderCompareSelector(runs) {
  if (!runs.length) {
    compareSelector.innerHTML = '<div class="empty-note">Run at least two jobs to compare them.</div>';
    return;
  }
  compareSelector.innerHTML = runs
    .slice(0, 12)
    .map(
      (run) => `
        <div class="check-row ${String(run.run_id) === String(selectedSavedRunId) ? "selected-row" : ""}">
          <label>
            <input type="checkbox" value="${run.run_id}" />
            <span>${run.strategy_name || "Strategy"} | ${run.run_id}</span>
          </label>
          <button class="inline-button" type="button" data-view-run-id="${run.run_id}">View</button>
        </div>
      `,
    )
    .join("");
  compareSelector.querySelectorAll("[data-view-run-id]").forEach((button) => {
    button.addEventListener("click", () => viewSavedRun(button.dataset.viewRunId, "backtestTab", "compareTab"));
  });
}

async function compareSelectedRuns() {
  const runIds = [...compareSelector.querySelectorAll('input[type="checkbox"]:checked')].map((input) => input.value);
  if (runIds.length < 2) {
    compareStatus.textContent = "Select at least two runs";
    return;
  }
  compareStatus.textContent = "Comparing…";
  const response = await fetch("/api/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ run_ids: runIds }),
  });
  const payload = await response.json();
  if (!response.ok) {
    compareStatus.textContent = payload.error || "Unable to compare";
    return;
  }
  renderCompareRows(payload.comparison?.runs || []);
  compareStatus.textContent = `${payload.comparison?.count || 0} run(s) compared`;
}

function renderCompareRows(rows) {
  if (!rows.length) {
    compareRows.innerHTML = '<tr><td colspan="6">No comparison yet</td></tr>';
    return;
  }
  compareRows.innerHTML = rows
    .map(
      (row) => `
        <tr>
          <td>${row.run_id}</td>
          <td>${row.strategy}</td>
          <td>${pct(row.median_return)}</td>
          <td>${pct(row.return_p05)}</td>
          <td>${pct(row.worst_drawdown)}</td>
          <td>${pct(row.robustness_score)}</td>
        </tr>
      `,
    )
    .join("");
}

function drawPathChart(preview) {
  const real = preview.real_close || [];
  const generated = preview.synthetic_paths || [];
  const series = [
    { label: "Real", values: real, color: themeColor("text"), width: 2.5 },
    ...generated.map((path, index) => ({
      label: `Path ${path.path_id}`,
      values: path.close,
      color: palette(index),
      width: 1.4,
    })),
  ];
  drawLineSeries(pathChart, series);
}

function drawReturnChart(preview) {
  drawHistogram(returnChart, [
    { label: "Real", data: preview.real_returns || { x: [], y: [] }, color: alphaColor("text", 0.62) },
    { label: "Synthetic", data: preview.synthetic_returns || { x: [], y: [] }, color: alphaColor("positive", 0.52) },
  ]);
}

function drawEquityChart(backtest) {
  const series = (backtest.equity_preview || []).map((path, index) => ({
    label: `Path ${path.path_id}`,
    values: path.equity,
    color: palette(index),
    width: 1.5,
  }));
  drawLineSeries(equityChart, series);
}

function redrawCharts() {
  if (!lastJob) {
    return;
  }
  if (lastJob.preview && Object.keys(lastJob.preview).length) {
    drawPathChart(lastJob.preview);
    drawReturnChart(lastJob.preview);
  }
  if (lastJob.backtest && Object.keys(lastJob.backtest).length) {
    renderBacktest(lastJob.backtest);
  }
}

function drawLineSeries(canvas, series) {
  const ctx = fitCanvas(canvas);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  ctx.clearRect(0, 0, width, height);
  const values = series.flatMap((item) => item.values || []);
  if (!values.length) {
    drawEmpty(canvas, "Waiting for data");
    return;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = 28;
  drawGrid(ctx, width, height, pad);
  series.forEach((item) => {
    const itemValues = item.values || [];
    if (!itemValues.length) {
      return;
    }
    ctx.beginPath();
    ctx.strokeStyle = item.color;
    ctx.lineWidth = item.width;
    itemValues.forEach((value, index) => {
      const x = pad + (index / Math.max(itemValues.length - 1, 1)) * (width - pad * 2);
      const y = height - pad - ((value - min) / Math.max(max - min, 1e-9)) * (height - pad * 2);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  });
}

function drawHistogram(canvas, series) {
  const ctx = fitCanvas(canvas);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  ctx.clearRect(0, 0, width, height);
  const allX = series.flatMap((item) => item.data?.x || []);
  const allY = series.flatMap((item) => item.data?.y || []);
  if (!allX.length || !allY.length) {
    drawEmpty(canvas, "Waiting for data");
    return;
  }
  const minX = Math.min(...allX);
  const maxX = Math.max(...allX);
  const maxY = Math.max(...allY);
  const pad = 28;
  drawGrid(ctx, width, height, pad);
  series.forEach((item, itemIndex) => {
    const xValues = item.data?.x || [];
    const yValues = item.data?.y || [];
    ctx.fillStyle = item.color;
    xValues.forEach((xValue, index) => {
      const x = pad + ((xValue - minX) / Math.max(maxX - minX, 1e-9)) * (width - pad * 2);
      const barHeight = (yValues[index] / Math.max(maxY, 1e-9)) * (height - pad * 2);
      const barWidth = Math.max(2, (width - pad * 2) / Math.max(xValues.length, 1) / 1.4);
      ctx.fillRect(x - barWidth / 2 + itemIndex * 1.5, height - pad - barHeight, barWidth, barHeight);
    });
  });
}

function drawGrid(ctx, width, height, pad) {
  ctx.strokeStyle = themeColor("grid");
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = pad + (i / 4) * (height - pad * 2);
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
  }
}

function drawEmpty(canvas, text) {
  const ctx = fitCanvas(canvas);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = themeColor("muted");
  const rootStyle = getComputedStyle(document.documentElement);
  const fontSize = rootStyle.getPropertyValue("--text-sm").trim() || "0.8125rem";
  const fontFamily = rootStyle.getPropertyValue("--font-body").trim() || "sans-serif";
  ctx.font = `${fontSize} ${fontFamily}`;
  ctx.textAlign = "center";
  ctx.fillText(text, width / 2, height / 2);
}

function fitCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 1;
  const height = canvas.clientHeight || 1;
  if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
    canvas.width = width * ratio;
    canvas.height = height * ratio;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return ctx;
}

function resetOutputs() {
  metricStatus.textContent = "-";
  metricKs.textContent = "-";
  metricVol.textContent = "-";
  metricMemo.textContent = "-";
  btRobustness.textContent = "-";
  btMedian.textContent = "-";
  btProfitable.textContent = "-";
  btDrawdown.textContent = "-";
  worstPathRows.innerHTML = '<tr><td colspan="6">-</td></tr>';
  outputHint.textContent = "No files yet";
  downloadActions.innerHTML = "";
  warningList.innerHTML = "";
  plotGrid.innerHTML = "";
  pathHint.textContent = "Running";
  drawAllEmpty("Training in progress");
}

function drawAllEmpty(text = "Waiting for data") {
  [pathChart, returnChart, equityChart, strategyReturnChart, drawdownChart].forEach((canvas) => drawEmpty(canvas, text));
}

function setBusy(isBusy) {
  runButton.disabled = isBusy;
  runButton.setAttribute("aria-busy", String(isBusy));
  runButton.textContent = isBusy ? "Running…" : "Run model";
}

function setStatus(status) {
  statusBadge.className = `status-badge ${status || "idle"}`;
  statusBadge.textContent = labelize(status || "idle");
}

function setMessage(message) {
  runMessage.textContent = message;
}

function fmt(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  if (Math.abs(number) >= 10) {
    return number.toFixed(1);
  }
  if (Math.abs(number) >= 1) {
    return number.toFixed(2);
  }
  return number.toFixed(4);
}

function pct(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  return `${(number * 100).toFixed(1)}%`;
}

function money(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  return `$${number.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function labelize(value) {
  return String(value || "idle")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function palette(index) {
  return themeColor(`series-${(index % 8) + 1}`);
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("synthmarket-theme", theme);
  themeIcon.textContent = theme === "dark" ? "Light" : "Dark";
  themeToggle.setAttribute("aria-label", theme === "dark" ? "Switch to light mode" : "Switch to dark mode");
  window.dispatchEvent(new CustomEvent("synthmarket-theme-change", { detail: { theme } }));
}

function themeColor(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(`--chart-${name}`).trim();
}

function alphaColor(name, alpha) {
  const color = themeColor(name);
  if (color.startsWith("oklch(") && color.endsWith(")")) {
    return `${color.slice(0, -1)} / ${alpha})`;
  }
  return color;
}

window.addEventListener("resize", redrawCharts);

setActiveTab(document.querySelector(".tab-button.active")?.dataset.tabTarget || "dataTab");
bootstrap();
