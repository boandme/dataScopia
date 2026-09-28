// When this page is served BY the API server itself (the normal case - see run.py), the
// API lives on the same origin the page was loaded from, so a relative base ('') is both
// correct and portable to whatever port the launcher picked. The explicit localhost:8000
// fallback only matters if this file is opened directly as file:// during development,
// without the server serving it.
const API_BASE = window.KAGGLE_API_BASE
  || new URLSearchParams(window.location.search).get('api')
  || (window.location.protocol === 'file:' ? 'http://localhost:8000' : '');
const state = {
  dataset: null,
  analysis: null,
  slug: null,
  findings: [],
  transformLog: [],          // [{type, params, description}] - replayed by the notebook export
  projectName: null,         // whatever the user last named this analysis - export titles default to it
};

const searchInput = document.getElementById('search-input');
const resultsEl = document.getElementById('results');
const resultsCountEl = document.getElementById('results-count');
const uploadInput = document.getElementById('upload-input');
const uploadBtn = document.getElementById('upload-btn');
const datasetInfoEl = document.getElementById('dataset-info');
const aboutDatasetEl = document.getElementById('about-dataset');
const aboutDatasetBodyEl = document.getElementById('about-dataset-body');
const sampleBannerEl = document.getElementById('sample-banner');
const fullAnalysisBtn = document.getElementById('full-analysis-btn');
const downloadFullBtn = document.getElementById('download-full-btn');
const statsContainer = document.getElementById('stats-container');
const insightsList = document.getElementById('insights-list');
const questionsList = document.getElementById('questions-list');
const columnExplorerSelect = document.getElementById('column-explorer-select');
const columnExplorerBody = document.getElementById('column-explorer-body');
const vizSectionEl = document.getElementById('viz-section');
const workspaceEmptyHintEl = document.getElementById('workspace-empty-hint');
function setVizSectionVisible(visible) {
  vizSectionEl.hidden = !visible;
  workspaceEmptyHintEl.hidden = visible;
}
const editorEl = document.getElementById('visualization-editor');
const chartTemplate = document.getElementById('chart-card-template');
const addChartBtn = document.getElementById('add-chart-btn');
const findingsEl = document.getElementById('findings');
const findingsListEl = document.getElementById('findings-list');
const questionInput = document.getElementById('question-input');
const questionAnswer = document.getElementById('question-answer');
const datasetEditorTemplate = document.getElementById('dataset-editor-template');
const addDatasetEditorBtn = document.getElementById('add-dataset-editor-btn');
const resetTransformBtn = document.getElementById('reset-transforms-btn');
const transformLogEl = document.getElementById('transform-log');
const saveAnalysisBtn = document.getElementById('save-analysis-btn');
const pastAnalysesPanel = document.getElementById('past-analyses-panel');
const pastAnalysesListEl = document.getElementById('past-analyses-list');
const saveNameOverlay = document.getElementById('save-name-overlay');
const saveNameInput = document.getElementById('save-name-input');
let chartSlots = [];
let datasetEditorSlots = [];

const chartTypes = [
  ['histogram', 'Histogram'], ['scatter', 'Scatter'], ['bar', 'Bar'],
  ['box', 'Box plot'], ['line', 'Line'], ['timeseries', 'Time series'], ['area', 'Area'],
  ['violin', 'Violin'], ['heatmap', 'Correlation matrix'], ['stacked_bar', 'Stacked bar'],
  ['pie', 'Pie'], ['distribution', 'Distribution by group'], ['target', 'Target vs feature'],
  ['treemap', 'Treemap'], ['map', 'Map'], ['missing', 'Missing values'], ['pair', 'Pair plot'],
];
const granularities = [
  ['none', 'Automatic'], ['day', 'Day'], ['week', 'Week'], ['month', 'Month'], ['quarter', 'Quarter'], ['year', 'Year'],
];
const FAMILY_LABELS = {
  time: 'Time', geo: 'Geography', distribution: 'Distribution', relationship: 'Relationship',
  comparison: 'Comparison', composition: 'Composition', target: 'Target', quality: 'Quality',
};
const aggregations = [
  ['none', 'Raw values'], ['count', 'Count'], ['sum', 'Sum'],
  ['mean', 'Average'], ['median', 'Median'], ['max', 'Max'], ['min', 'Min'],
];
const sortOptions = [['none', 'Default order'], ['desc', 'Sort descending'], ['asc', 'Sort ascending']];

// Trimmed modebar (default Plotly ships ~11 buttons, which overlap badly in a narrow
// card) plus no logo, so the remaining zoom/pan/reset/download controls have room.
const PLOTLY_CONFIG = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines', 'hoverClosestCartesian', 'hoverCompareCartesian'],
  toImageButtonOptions: {format: 'png', scale: 2},
};
const PLOTLY_LAYOUT_BASE = {
  margin: {t: 44, r: 20, l: 56, b: 46},
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: {color: '#e5e7eb'},
  legend: {orientation: 'h', y: -0.2},
};

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}

function boldify(escapedText) {
  return escapedText.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
}

// Kaggle descriptions are third-party, untrusted text. Everything is escaped via
// escapeHtml() first; the only real HTML tags inserted afterward are our own literal
// wrappers (h1-6/ul/li/p/strong), never derived from the input, so this is safe against
// script injection even though we render it into the DOM.
function formatDescription(text) {
  if (!text) return '<p>No description provided.</p>';
  const lines = text.split('\n');
  const html = [];
  let inList = false;
  const closeList = () => { if (inList) { html.push('</ul>'); inList = false; } };
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) { closeList(); continue; }
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      closeList();
      const level = Math.min(heading[1].length + 3, 6);
      html.push(`<h${level} style="margin: 12px 0 4px; font-size: 1rem;">${escapeHtml(heading[2])}</h${level}>`);
      continue;
    }
    const bullet = line.match(/^[-*]\s+(.*)$/);
    if (bullet) {
      if (!inList) { html.push('<ul style="margin: 4px 0; padding-left: 20px;">'); inList = true; }
      html.push(`<li>${boldify(escapeHtml(bullet[1]))}</li>`);
      continue;
    }
    closeList();
    html.push(`<p style="margin: 6px 0;">${boldify(escapeHtml(line))}</p>`);
  }
  closeList();
  return html.join('');
}

function fieldOptions(fields, includeBlank = false, blankLabel = 'None', descriptions = {}) {
  const blank = includeBlank ? `<option value="">${blankLabel}</option>` : '';
  const opts = fields
    .map((field) => {
      const desc = descriptions[field];
      const titleAttr = desc ? ` title="${escapeHtml(desc)}"` : '';
      return `<option value="${field}"${titleAttr}>${field}</option>`;
    })
    .join('');
  return blank + opts;
}

function aggregateRows(rows, groupField, valueField, method) {
  const groups = new Map();
  rows.forEach((row) => {
    const key = row[groupField];
    if (key === null || key === undefined || key === '') return;
    if (!groups.has(key)) groups.set(key, []);
    if (method === 'count') {
      groups.get(key).push(1);
      return;
    }
    const v = Number(row[valueField]);
    if (!Number.isNaN(v)) groups.get(key).push(v);
  });
  return [...groups.entries()].map(([key, values]) => {
    let value;
    if (!values.length) value = 0;
    else if (method === 'count') value = values.length;
    else if (method === 'sum') value = values.reduce((a, b) => a + b, 0);
    else if (method === 'mean') value = values.reduce((a, b) => a + b, 0) / values.length;
    else if (method === 'median') {
      const sorted = [...values].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      value = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    } else if (method === 'max') value = Math.max(...values);
    else if (method === 'min') value = Math.min(...values);
    else value = values[0];
    return {key, value};
  });
}

function applySort(entries, order) {
  if (order === 'asc') return [...entries].sort((a, b) => a.value - b.value);
  if (order === 'desc') return [...entries].sort((a, b) => b.value - a.value);
  return entries;
}

function plotlyTypeFor(chartType) {
  if (chartType === 'line' || chartType === 'area') return 'scatter';
  if (chartType === 'stacked_bar') return 'bar';
  return chartType;
}

function chartConfigOf(slot) {
  return {
    type: slot.type.value, x: slot.x.value, y: slot.y.value, agg: slot.agg.value,
    group: slot.group.value, sort: slot.sort.value, granularity: slot.granularity.value,
    customTitle: slot.titleInput.value.trim(),
  };
}

// Charts are drawn by the server from the FULL cached dataframe (the browser only holds a
// 300-row preview). The old in-browser renderer below stays as a fallback for when the
// server-side copy has expired or the request fails, so a chart never just disappears.
async function renderChart(slot) {
  if (!state.analysis) return;
  window.markProjectDirty?.();
  const token = (slot.renderToken = (slot.renderToken || 0) + 1);
  if (state.dataset?.dataset_id) {
    try {
      const response = await fetch(`${API_BASE}/charts/render`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({dataset_id: state.dataset.dataset_id, config: chartConfigOf(slot)}),
      });
      if (token !== slot.renderToken) return; // a newer request superseded this one
      if (response.ok) {
        drawServerChart(slot, await response.json());
        return;
      }
    } catch (error) { /* fall through to the preview renderer */ }
  }
  if (token !== slot.renderToken) return;
  slot.notesEl.hidden = true;
  renderChartFromPreview(slot);
}

function drawServerChart(slot, result) {
  const meta = result.meta || {};
  const title = meta.title && meta.title.length > 60 ? `${meta.title.slice(0, 59)}…` : (meta.title || '');
  Plotly.react(
    slot.area,
    result.data || [],
    {...PLOTLY_LAYOUT_BASE, ...(result.layout || {}), title: {text: title, x: 0.02, xanchor: 'left', font: {size: 14}}},
    PLOTLY_CONFIG
  );
  const notes = [...(meta.notes || [])];
  if (meta.error) notes.unshift(meta.error);
  slot.notesEl.hidden = notes.length === 0;
  slot.notesEl.innerHTML = notes.map((note) => `<p>${escapeHtml(note)}</p>`).join('');
  slot.card.dataset.effectiveType = meta.effective_type || slot.type.value;
  slot.serverTitle = meta.title || '';
  updateGranularityVisibility(slot);
}

function updateGranularityVisibility(slot) {
  const timeLike = ['line', 'area', 'timeseries'].includes(slot.type.value);
  slot.granularityLabel.hidden = !timeLike;
}

function renderChartFromPreview(slot) {
  const rows = state.analysis.preview || [];
  const type = slot.type.value;
  const x = slot.x.value;
  const y = slot.y.value;
  const agg = slot.agg.value;
  const groupField = slot.group.value;
  const sortOrder = slot.sort.value;
  const customTitle = slot.titleInput.value.trim();
  let traces = [];
  let title = customTitle;

  if (agg !== 'none' && x && type !== 'heatmap') {
    const plotlyType = plotlyTypeFor(type);
    const buildTrace = (subset, name) => {
      let entries = aggregateRows(subset, x, y, agg);
      entries = applySort(entries, sortOrder);
      return {
        type: plotlyType === 'pie' ? 'pie' : plotlyType,
        mode: type === 'line' ? 'lines+markers' : undefined,
        fill: type === 'area' ? 'tozeroy' : undefined,
        x: plotlyType === 'pie' ? undefined : entries.map((e) => e.key),
        y: plotlyType === 'pie' ? undefined : entries.map((e) => e.value),
        labels: plotlyType === 'pie' ? entries.map((e) => e.key) : undefined,
        values: plotlyType === 'pie' ? entries.map((e) => e.value) : undefined,
        name,
      };
    };
    if (groupField && groupField !== x) {
      const groupValues = [...new Set(rows.map((r) => r[groupField]).filter((v) => v !== null && v !== undefined && v !== ''))].slice(0, 8);
      traces = groupValues.map((gv) => buildTrace(rows.filter((r) => r[groupField] === gv), String(gv)));
      title = title || `${agg} of ${agg === 'count' ? 'rows' : y} by ${x}, grouped by ${groupField}`;
    } else {
      traces = [buildTrace(rows, `${agg} of ${agg === 'count' ? 'rows' : y}`)];
      title = title || `${agg} of ${agg === 'count' ? 'rows' : y} by ${x}`;
    }
  } else {
    const values = rows.map((row) => row[x]).filter((value) => value !== null && value !== undefined);
    if (type === 'histogram') {
      traces = [{type, x: values, name: x}];
      title = title || `Distribution of ${x}`;
    } else if (type === 'pie') {
      const counts = {};
      values.forEach((value) => { counts[value] = (counts[value] || 0) + 1; });
      traces = [{type: 'pie', labels: Object.keys(counts), values: Object.values(counts)}];
      title = title || `Share of ${x}`;
    } else if (type === 'box' || type === 'violin') {
      traces = [{type, x: rows.map((row) => row[x]), y: rows.map((row) => row[y]), name: `${y} by ${x}`}];
      title = title || `${y} by ${x}`;
    } else if (type === 'heatmap') {
      const numeric = state.analysis.numeric_columns || [];
      const matrix = numeric.map((a) => numeric.map((b) => state.analysis.relationship_summary.find((r) => (r.column_a === a && r.column_b === b) || (r.column_a === b && r.column_b === a))?.correlation ?? (a === b ? 1 : 0)));
      traces = [{type, z: matrix, x: numeric, y: numeric, colorscale: 'Viridis'}];
      title = title || 'Correlation heatmap';
    } else if (type === 'bar' || type === 'stacked_bar') {
      const counts = {};
      values.forEach((value) => { counts[value] = (counts[value] || 0) + 1; });
      traces = [{type: 'bar', x: Object.keys(counts), y: Object.values(counts), name: x}];
      title = title || `Counts by ${x}`;
    } else {
      const scatterType = plotlyTypeFor(type);
      if (groupField) {
        const groupValues = [...new Set(rows.map((r) => r[groupField]).filter((v) => v !== null && v !== undefined && v !== ''))].slice(0, 8);
        traces = groupValues.map((gv) => {
          const subset = rows.filter((r) => r[groupField] === gv);
          return {
            type: scatterType, mode: type === 'line' ? 'lines+markers' : (type === 'scatter' ? 'markers' : undefined),
            fill: type === 'area' ? 'tozeroy' : undefined,
            x: subset.map((row) => row[x]), y: subset.map((row) => row[y]), name: String(gv),
          };
        });
      } else {
        // x/y must come from the same filtered row set: independently filtering x (to
        // drop nulls) while leaving y unfiltered silently misaligns x[i]/y[i] the moment
        // any row is missing an x value, scrambling the plotted relationship.
        const pairedX = [];
        const pairedY = [];
        rows.forEach((row) => {
          const xv = row[x];
          const yv = row[y];
          if (xv === null || xv === undefined || yv === null || yv === undefined) return;
          pairedX.push(xv);
          pairedY.push(yv);
        });
        traces = [{type: scatterType, mode: type === 'line' ? 'lines+markers' : (type === 'scatter' ? 'markers' : undefined), x: pairedX, y: pairedY, fill: type === 'area' ? 'tozeroy' : undefined, name: `${x} vs ${y}`}];
      }
      title = title || `${x} vs ${y}`;
    }
  }

  // Force a true linear (proportional) scale on any axis plotting a known-numeric column,
  // rather than trusting Plotly's auto-detected axis type. Auto-detection can pick
  // 'category' (equal spacing regardless of value) for low-cardinality numeric columns,
  // which flattens or distorts the real shape of a relationship - most visible, and most
  // damaging, on scatter/line/area plots where the whole point is reading that shape.
  const axisLayout = {};
  if (type === 'scatter' || type === 'line' || type === 'area') {
    const numericCols = state.analysis.numeric_columns || [];
    if (numericCols.includes(x)) axisLayout.xaxis = {type: 'linear'};
    if (numericCols.includes(y)) axisLayout.yaxis = {type: 'linear'};
  }

  const shortTitle = title && title.length > 42 ? `${title.slice(0, 41)}…` : title;
  Plotly.react(
    slot.area,
    traces,
    {...PLOTLY_LAYOUT_BASE, ...axisLayout, title: {text: shortTitle, x: 0.02, xanchor: 'left', font: {size: 14}}},
    PLOTLY_CONFIG
  );
}

// Plotly measures its container's width at the moment Plotly.react() runs. When several
// cards are created in the same grid back-to-back (e.g. rebuilding 2 charts + 1 dataset
// editor), the first one can be measured while it's still the grid's only item - full row
// width - before the later siblings arrive and the grid recomputes to a narrower per-item
// width. The result: the first card renders oversized and everything else gets pushed
// below it. This forces a corrective resize pass, deferred two animation frames so it
// runs strictly after the browser has finished laying out the *final* grid, whenever the
// grid's item count changes (chart/editor added, removed, or a full rebuild).
function resizeAllCharts() {
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      chartSlots.forEach((slot) => {
        if (slot.area && slot.area.data) {
          try { Plotly.Plots.resize(slot.area); } catch (error) { /* not rendered yet */ }
        }
      });
    });
  });
}

function createChartSlot(label, fields, recommendation, savedConfig) {
  const node = chartTemplate.content.firstElementChild.cloneNode(true);
  node.querySelector('.chart-card-label').textContent = label;
  editorEl.appendChild(node);
  const slot = {
    card: node,
    label,
    type: node.querySelector('.chart-type'),
    x: node.querySelector('.chart-x'),
    y: node.querySelector('.chart-y'),
    agg: node.querySelector('.chart-agg'),
    group: node.querySelector('.chart-group'),
    sort: node.querySelector('.chart-sort'),
    granularity: node.querySelector('.chart-granularity'),
    granularityLabel: node.querySelector('.chart-granularity-label'),
    notesEl: node.querySelector('.chart-notes'),
    titleInput: node.querySelector('.chart-title-input'),
    area: node.querySelector('.chart-area'),
    saveBtn: node.querySelector('.chart-save-btn'),
    removeBtn: node.querySelector('.chart-remove-btn'),
    exportFormat: node.querySelector('.chart-export-format'),
    exportScale: node.querySelector('.chart-export-scale'),
    exportBtn: node.querySelector('.chart-export-btn'),
  };
  const columnDescriptions = {...(state.dataset?.inferred_column_descriptions || {}), ...(state.dataset?.column_descriptions || {})};
  slot.type.innerHTML = chartTypes.map(([value, text]) => `<option value="${value}">${text}</option>`).join('');
  slot.x.innerHTML = fieldOptions(fields, false, 'None', columnDescriptions);
  slot.y.innerHTML = fieldOptions(fields, true, 'None', columnDescriptions);
  slot.agg.innerHTML = aggregations.map(([value, text]) => `<option value="${value}">${text}</option>`).join('');
  slot.group.innerHTML = fieldOptions(fields, true, 'No grouping', columnDescriptions);
  slot.sort.innerHTML = sortOptions.map(([value, text]) => `<option value="${value}">${text}</option>`).join('');
  slot.granularity.innerHTML = granularities.map(([value, text]) => `<option value="${value}">${text}</option>`).join('');

  const cfg = savedConfig || recommendation || {};
  slot.type.value = chartTypes.some(([value]) => value === cfg.type) ? cfg.type : 'histogram';
  slot.x.value = cfg.x || fields[0] || '';
  slot.y.value = cfg.y || (cfg.type ? '' : (fields[1] || fields[0] || ''));
  slot.agg.value = cfg.agg || 'none';
  slot.group.value = cfg.group || '';
  slot.sort.value = cfg.sort || 'none';
  slot.granularity.value = cfg.granularity || 'none';
  slot.titleInput.value = cfg.customTitle || '';
  updateGranularityVisibility(slot);

  [slot.type, slot.x, slot.y, slot.agg, slot.group, slot.sort, slot.granularity].forEach((control) => control.addEventListener('change', () => {
    updateGranularityVisibility(slot);
    renderChart(slot);
  }));
  let titleTimer = null;
  slot.titleInput.addEventListener('input', () => {
    clearTimeout(titleTimer);
    titleTimer = setTimeout(() => renderChart(slot), 300);
  });
  slot.saveBtn.addEventListener('click', () => saveChartFinding(slot, label));
  slot.removeBtn.addEventListener('click', () => {
    node.remove();
    chartSlots = chartSlots.filter((s) => s !== slot);
    resizeAllCharts();
  });
  slot.exportBtn.addEventListener('click', () => exportChart(slot));
  renderChart(slot);
  chartSlots.push(slot);
  return slot;
}

function exportChart(slot) {
  if (typeof Plotly === 'undefined' || !slot.area || !slot.area.data) return;
  const format = slot.exportFormat.value || 'png';
  const scale = Number(slot.exportScale.value) || 2;
  const filename = `${(state.analysis?.dataset_name || 'chart').replace(/[^a-z0-9_-]+/gi, '_')}-${slot.x.value || 'chart'}`;
  Plotly.downloadImage(slot.area, {format, filename, scale}).catch(() => {
    slot.exportBtn.textContent = 'Export failed';
    setTimeout(() => { slot.exportBtn.textContent = 'Export chart'; }, 1500);
  });
}

function saveChartFinding(slot, label) {
  const chart = chartConfigOf(slot);
  const descriptor = chart.agg !== 'none'
    ? `${chart.agg} of ${chart.y || 'rows'} by ${chart.x}${chart.group ? ` (grouped by ${chart.group})` : ''}`
    : `${chart.type} of ${chart.x}${chart.y ? ` vs ${chart.y}` : ''}`;
  const title = chart.customTitle || `${label}: ${descriptor}`;
  state.findings.push({
    type: 'chart',
    title,
    summary: `${descriptor} for ${state.analysis?.dataset_name || 'the active dataset'}.`,
    chart,
  });
  renderFindings();
  const original = slot.saveBtn.textContent;
  slot.saveBtn.textContent = 'Saved ✓';
  setTimeout(() => { slot.saveBtn.textContent = original; }, 1200);
}

function rebuildChartSlots(fields, recommendations, savedConfigs) {
  editorEl.innerHTML = '';
  chartSlots = [];
  if (savedConfigs && savedConfigs.length) {
    savedConfigs.forEach((cfg, index) => createChartSlot(cfg.label || `Chart ${index + 1}`, fields, null, cfg));
  } else {
    // Start from the two best recommendations (they are ordered to differ in kind); the
    // gallery above offers the rest, and "+ Add chart" starts a blank one.
    const fallbackLabels = ['Primary relationship', 'Distribution or categories'];
    [0, 1].forEach((index) => {
      const rec = recommendations[index];
      createChartSlot(rec?.title || fallbackLabels[index], fields, rec);
    });
  }
  resizeAllCharts();
}

function renderRecommendations() {
  const recs = state.analysis?.recommendations || [];
  const panel = document.getElementById('recs-panel');
  const grid = document.getElementById('recs-grid');
  panel.hidden = recs.length === 0;
  grid.innerHTML = recs.map((rec, index) => `
    <div class="rec-card" data-family="${escapeHtml(rec.family || 'other')}">
      <span class="chip rec-family">${escapeHtml(FAMILY_LABELS[rec.family] || rec.family || 'Chart')}</span>
      <strong>${escapeHtml(rec.title || rec.type)}</strong>
      <p>${escapeHtml(rec.reason || '')}</p>
      <button type="button" class="btn secondary card-small-btn" data-rec-index="${index}">Add chart</button>
    </div>
  `).join('');
  grid.querySelectorAll('[data-rec-index]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const rec = recs[Number(btn.dataset.recIndex)];
      const fields = currentFieldNames();
      if (!rec || !fields.length) return;
      createChartSlot(rec.title || 'Recommended chart', fields, rec);
      resizeAllCharts();
      btn.textContent = 'Added ✓';
      setTimeout(() => { btn.textContent = 'Add chart'; }, 1200);
    });
  });
}

// ---- Dataset editors: filter/clean/transform the full server-side sample ----

const CATEGORY_META = {
  filter: {label: 'Filter', className: 'cat-filter'},
  clean: {label: 'Clean', className: 'cat-clean'},
  transform: {label: 'Transform', className: 'cat-transform'},
};

const TRANSFORM_OPS = [
  {
    type: 'filter', label: 'Filter rows', category: 'filter',
    fields: [
      {key: 'column', label: 'Column', kind: 'column'},
      {key: 'comparator', label: 'Comparator', kind: 'select', options: [['>', '>'], ['>=', '>='], ['<', '<'], ['<=', '<='], ['==', 'equals'], ['!=', 'not equals']]},
      {key: 'value', label: 'Value', kind: 'text'},
    ],
  },
  {
    type: 'drop_missing', label: 'Drop rows with missing values', category: 'clean',
    fields: [
      {key: 'column', label: 'Column', kind: 'column', allowBlank: true, blankLabel: 'Any column'},
    ],
  },
  {
    type: 'fill_missing', label: 'Fill missing values', category: 'clean',
    fields: [
      {key: 'column', label: 'Column', kind: 'column'},
      {key: 'strategy', label: 'Fill with', kind: 'select', options: [['mean', 'Mean'], ['median', 'Median'], ['mode', 'Mode'], ['constant', 'Constant value']]},
      {key: 'constant_value', label: 'Constant value', kind: 'text', showIf: {strategy: 'constant'}},
    ],
  },
  {
    type: 'drop_outliers', label: 'Drop outliers (IQR rule)', category: 'clean',
    fields: [{key: 'column', label: 'Numeric column', kind: 'column', numericOnly: true}],
  },
  {
    type: 'drop_duplicates', label: 'Remove duplicate rows', category: 'clean',
    fields: [],
  },
  {
    type: 'replace_missing_tokens', label: 'Treat placeholders (N/A, -, ?) as missing', category: 'clean',
    fields: [
      {key: 'column', label: 'Column', kind: 'column'},
      {key: 'values', label: 'Placeholder values (comma-separated)', kind: 'list', default: 'N/A, n/a, NA, -, ?, null, none'},
    ],
  },
  {
    type: 'standardize_text', label: 'Standardize spelling and spacing', category: 'clean',
    fields: [{key: 'column', label: 'Text column', kind: 'column'}],
  },
  {
    type: 'rename_column', label: 'Rename column', category: 'transform',
    fields: [
      {key: 'column', label: 'Column', kind: 'column'},
      {key: 'new_name', label: 'New name', kind: 'text'},
    ],
  },
  {
    type: 'cast_type', label: 'Change column type', category: 'transform',
    fields: [
      {key: 'column', label: 'Column', kind: 'column'},
      {key: 'target_type', label: 'New type', kind: 'select', options: [['numeric', 'Numeric'], ['string', 'Text'], ['datetime', 'Date/time'], ['category', 'Category']]},
    ],
  },
  {
    type: 'drop_column', label: 'Drop column', category: 'transform',
    fields: [{key: 'column', label: 'Column', kind: 'column'}],
  },
  {
    type: 'combine_columns', label: 'Combine columns (merge)', category: 'transform',
    fields: [
      {key: 'columns', label: 'Columns (ctrl/cmd-click for multiple)', kind: 'multi-column'},
      {key: 'operation', label: 'Combine using', kind: 'select', options: [['concat', 'Join as text'], ['sum', 'Sum'], ['mean', 'Average'], ['coalesce', 'First non-empty']]},
      {key: 'separator', label: 'Separator', kind: 'text', default: ' ', showIf: {operation: 'concat'}},
      {key: 'new_name', label: 'New column name', kind: 'text'},
    ],
  },
];

function currentFieldNames() {
  return Object.keys((state.analysis?.preview || [])[0] || {});
}

function buildTransformFields(slot) {
  const opConfig = TRANSFORM_OPS.find((o) => o.type === slot.opType.value);
  slot.fieldsEl.innerHTML = '';
  slot.fieldInputs = {};
  slot.fieldWrappers = {};
  if (!opConfig) return;

  const updateVisibility = () => {
    opConfig.fields.forEach((fieldSpec) => {
      if (!fieldSpec.showIf) return;
      const [depKey, depVal] = Object.entries(fieldSpec.showIf)[0];
      const depInput = slot.fieldInputs[depKey];
      slot.fieldWrappers[fieldSpec.key].hidden = !(depInput && depInput.value === depVal);
    });
  };

  opConfig.fields.forEach((fieldSpec) => {
    const wrapper = document.createElement('label');
    wrapper.textContent = fieldSpec.label;
    let input;
    if (fieldSpec.kind === 'column' || fieldSpec.kind === 'multi-column') {
      const cols = fieldSpec.numericOnly ? (state.analysis?.numeric_columns || []) : currentFieldNames();
      input = document.createElement('select');
      if (fieldSpec.kind === 'multi-column') { input.multiple = true; input.size = Math.min(6, Math.max(2, cols.length)); }
      input.innerHTML = (fieldSpec.allowBlank ? `<option value="">${fieldSpec.blankLabel || 'None'}</option>` : '') + cols.map((c) => `<option value="${c}">${c}</option>`).join('');
    } else if (fieldSpec.kind === 'select') {
      input = document.createElement('select');
      input.innerHTML = fieldSpec.options.map(([v, l]) => `<option value="${v}">${l}</option>`).join('');
    } else {
      input = document.createElement('input');
      input.type = 'text';
      if (fieldSpec.default) input.value = fieldSpec.default;
    }
    input.addEventListener('change', updateVisibility);
    wrapper.appendChild(input);
    slot.fieldsEl.appendChild(wrapper);
    slot.fieldInputs[fieldSpec.key] = input;
    slot.fieldWrappers[fieldSpec.key] = wrapper;
  });
  updateVisibility();
}

function refreshDatasetEditorFieldOptions(slot) {
  const opConfig = TRANSFORM_OPS.find((o) => o.type === slot.opType.value);
  if (!opConfig) return;
  const fields = currentFieldNames();
  opConfig.fields.forEach((fieldSpec) => {
    const input = slot.fieldInputs[fieldSpec.key];
    if (!input || (fieldSpec.kind !== 'column' && fieldSpec.kind !== 'multi-column')) return;
    const cols = fieldSpec.numericOnly ? (state.analysis?.numeric_columns || []) : fields;
    const prevSelected = fieldSpec.kind === 'multi-column'
      ? [...input.selectedOptions].map((o) => o.value)
      : [input.value];
    input.innerHTML = (fieldSpec.allowBlank ? `<option value="">${fieldSpec.blankLabel || 'None'}</option>` : '') + cols.map((c) => `<option value="${c}">${c}</option>`).join('');
    [...input.options].forEach((opt) => { if (prevSelected.includes(opt.value)) opt.selected = true; });
  });
}

function updateEditorBadge(slot) {
  const opConfig = TRANSFORM_OPS.find((o) => o.type === slot.opType.value);
  const meta = CATEGORY_META[opConfig?.category] || CATEGORY_META.transform;
  slot.badgeEl.className = `type-badge ${meta.className}`;
  slot.badgeEl.textContent = meta.label;
}

function createDatasetEditorSlot(label) {
  // Appended into the same grid as chart cards (not a separate container), so both card
  // types are always identically sized and dataset editors naturally land below the
  // charts already in the grid rather than in their own section above them.
  const node = datasetEditorTemplate.content.firstElementChild.cloneNode(true);
  node.querySelector('.dataset-editor-label').textContent = label;
  editorEl.appendChild(node);
  const slot = {
    card: node,
    opType: node.querySelector('.de-op-type'),
    fieldsEl: node.querySelector('.de-fields'),
    applyBtn: node.querySelector('.de-apply-btn'),
    resultEl: node.querySelector('.de-result'),
    removeBtn: node.querySelector('.de-remove-btn'),
    badgeEl: node.querySelector('.de-type-badge'),
    impactEl: node.querySelector('.de-impact'),
    fieldInputs: {},
    fieldWrappers: {},
  };
  slot.opType.innerHTML = TRANSFORM_OPS.map((o) => `<option value="${o.type}">${o.label}</option>`).join('');
  slot.opType.addEventListener('change', () => { buildTransformFields(slot); updateEditorBadge(slot); });
  buildTransformFields(slot);
  updateEditorBadge(slot);
  slot.applyBtn.addEventListener('click', () => applyDatasetTransform(slot));
  slot.removeBtn.addEventListener('click', () => {
    node.remove();
    datasetEditorSlots = datasetEditorSlots.filter((s) => s !== slot);
    resizeAllCharts();
  });
  datasetEditorSlots.push(slot);
  return slot;
}

function resetDatasetEditorSlots() {
  // Remove only the tracked dataset-editor cards, not the whole (shared) grid.
  datasetEditorSlots.forEach((slot) => slot.card.remove());
  datasetEditorSlots = [];
  createDatasetEditorSlot('Dataset editor 1');
  resizeAllCharts();
}

function renderImpact(slot, beforeRows, afterRows) {
  if (typeof beforeRows !== 'number' || typeof afterRows !== 'number' || beforeRows <= 0) {
    slot.impactEl.hidden = true;
    return;
  }
  const removed = beforeRows - afterRows;
  const pctKept = Math.max(0, Math.min(100, (afterRows / beforeRows) * 100));
  slot.impactEl.hidden = false;
  slot.impactEl.innerHTML = `
    <div class="impact-bar"><div class="impact-bar-fill" style="width: ${pctKept}%;"></div></div>
    <div class="impact-meta">
      <span>${afterRows.toLocaleString()} of ${beforeRows.toLocaleString()} rows kept</span>
      <span>${removed > 0 ? `−${removed.toLocaleString()} rows` : 'no rows removed'}</span>
    </div>
  `;
}

async function applyDatasetTransform(slot) {
  if (!state.analysis || !state.dataset?.dataset_id) {
    slot.resultEl.textContent = 'Load a dataset first.';
    slot.resultEl.style.color = 'var(--danger)';
    return;
  }
  const opConfig = TRANSFORM_OPS.find((o) => o.type === slot.opType.value);
  if (!opConfig) return;
  const params = {};
  opConfig.fields.forEach((fieldSpec) => {
    const input = slot.fieldInputs[fieldSpec.key];
    if (!input) return;
    if (fieldSpec.kind === 'multi-column') params[fieldSpec.key] = [...input.selectedOptions].map((o) => o.value);
    else if (fieldSpec.kind === 'list') params[fieldSpec.key] = input.value.split(',').map((v) => v.trim()).filter(Boolean);
    else params[fieldSpec.key] = input.value;
  });

  const beforeRows = state.analysis.research_snapshot?.rows ?? state.analysis.shape?.rows;

  slot.applyBtn.disabled = true;
  slot.applyBtn.textContent = 'Applying...';
  slot.resultEl.textContent = '';
  try {
    const outcome = await performTransform(opConfig.type, params);
    if (!outcome.ok) {
      slot.resultEl.textContent = outcome.message;
      slot.resultEl.style.color = 'var(--danger)';
      return;
    }
    slot.resultEl.textContent = `✓ ${outcome.result.description}`;
    slot.resultEl.style.color = 'var(--accent-2)';
    renderImpact(slot, beforeRows, outcome.result.shape?.rows);
    refreshDatasetEditorFieldOptions(slot);
  } finally {
    slot.applyBtn.disabled = false;
    slot.applyBtn.textContent = 'Apply';
  }
}

// One code path for every transform - dataset editors and the data-quality "Fix" buttons -
// so the transform log the notebook export replays is always complete and in order.
async function performTransform(type, params) {
  try {
    const response = await fetch(`${API_BASE}/datasets/transform`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        dataset_id: state.dataset.dataset_id,
        dataset_name: state.analysis.dataset_name,
        type,
        params,
      }),
    });
    const result = await response.json();
    if (!response.ok) return {ok: false, message: result.detail || 'Transform failed.'};
    state.analysis = result.analysis;
    state.transformLog.push({type, params, description: result.description});
    resetTransformBtn.hidden = false;
    datasetEditorSlots.forEach((s) => refreshDatasetEditorFieldOptions(s));
    renderTransformLog();
    renderAnalysis({rebuildCharts: false});
    return {ok: true, result};
  } catch (error) {
    return {ok: false, message: 'Transform failed. Is the API server running?'};
  }
}

function renderTransformLog() {
  window.markProjectDirty?.();
  if (!state.transformLog.length) {
    transformLogEl.hidden = true;
    transformLogEl.innerHTML = '';
    return;
  }
  transformLogEl.hidden = false;
  transformLogEl.innerHTML = state.transformLog.map((entry) => `<li>${escapeHtml(typeof entry === 'string' ? entry : entry.description)}</li>`).join('');
}

async function resetAllTransforms() {
  if (!state.dataset?.dataset_id) return;
  resetTransformBtn.disabled = true;
  try {
    const response = await fetch(`${API_BASE}/datasets/transform/reset`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({dataset_id: state.dataset.dataset_id, dataset_name: state.analysis?.dataset_name}),
    });
    const result = await response.json();
    if (!response.ok) return;
    state.analysis = result.analysis;
    state.transformLog = [];
    renderTransformLog();
    resetTransformBtn.hidden = true;
    datasetEditorSlots.forEach((s) => refreshDatasetEditorFieldOptions(s));
    renderAnalysis({rebuildCharts: false});
  } finally {
    resetTransformBtn.disabled = false;
  }
}

function refreshChartFieldOptions(slot, fields) {
  const columnDescriptions = {...(state.dataset?.inferred_column_descriptions || {}), ...(state.dataset?.column_descriptions || {})};
  const keepX = fields.includes(slot.x.value) ? slot.x.value : (fields[0] || '');
  const keepY = fields.includes(slot.y.value) ? slot.y.value : '';
  const keepGroup = fields.includes(slot.group.value) ? slot.group.value : '';
  slot.x.innerHTML = fieldOptions(fields, false, 'None', columnDescriptions);
  slot.y.innerHTML = fieldOptions(fields, true, 'None', columnDescriptions);
  slot.group.innerHTML = fieldOptions(fields, true, 'No grouping', columnDescriptions);
  slot.x.value = keepX;
  slot.y.value = keepY;
  slot.group.value = keepGroup;
}

function renderCappedList(el, items, cap, emptyMessage) {
  if (!items || !items.length) {
    el.innerHTML = `<li>${escapeHtml(emptyMessage)}</li>`;
    return;
  }
  const visible = items.slice(0, cap);
  const rest = items.slice(cap);
  const groupClass = `capped-${Math.random().toString(36).slice(2, 8)}`;
  let html = visible.map((item) => `<li>${escapeHtml(item)}</li>`).join('');
  if (rest.length) {
    html += rest.map((item) => `<li class="${groupClass}" hidden>${escapeHtml(item)}</li>`).join('');
    html += `<li style="border: none; padding-top: 4px;"><button type="button" class="insights-toggle">+${rest.length} more</button></li>`;
  }
  el.innerHTML = html;
  if (rest.length) {
    const toggleBtn = el.querySelector('.insights-toggle');
    toggleBtn.addEventListener('click', () => {
      const hiddenItems = el.querySelectorAll(`.${groupClass}`);
      const isHidden = hiddenItems[0]?.hidden;
      hiddenItems.forEach((li) => { li.hidden = !isHidden; });
      toggleBtn.textContent = isHidden ? 'Show less' : `+${rest.length} more`;
    });
  }
}

// ---- Data quality report: scored audit with one-click fixes that run through the same
// transform pipeline as the dataset editors (so they are logged and reproduced in the notebook).

const FIX_LABELS = {
  drop_column: 'Drop column', drop_duplicates: 'Remove duplicates', replace_missing_tokens: 'Mark as missing',
  standardize_text: 'Standardize', cast_type: 'Convert type', drop_outliers: 'Remove outliers',
  filter: 'Apply filter', rename_column: 'Rename',
};
const SEVERITY_ORDER = {high: 0, medium: 1, low: 2};

function renderQualityPanel() {
  const panel = document.getElementById('quality-panel');
  const report = state.analysis?.data_quality;
  if (!report) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const scoreClass = report.score >= 85 ? 'q-good' : (report.score >= 65 ? 'q-ok' : 'q-poor');
  document.getElementById('quality-score').innerHTML = `
    <span class="q-number ${scoreClass}">${report.score}</span><span class="q-of">/100</span>
    <span class="q-label">${escapeHtml(report.label)}</span>
    <span class="chip sev-high">${report.counts.high} high</span>
    <span class="chip sev-medium">${report.counts.medium} medium</span>
    <span class="chip sev-low">${report.counts.low} low</span>`;
  const issues = [...(report.issues || [])].sort((a, b) => (SEVERITY_ORDER[a.severity] ?? 3) - (SEVERITY_ORDER[b.severity] ?? 3));
  const host = document.getElementById('quality-issues');
  if (!issues.length) {
    host.innerHTML = '<p class="hint">No data-quality issues detected.</p>';
    return;
  }
  const VISIBLE = 5;
  host.innerHTML = issues.map((issue, index) => {
    const fix = issue.fix;
    const examples = (issue.examples || []).length ? `<div class="q-examples">e.g. ${issue.examples.map((e) => `<code>${escapeHtml(e)}</code>`).join(' ')}</div>` : '';
    return `
      <div class="q-issue sev-border-${issue.severity}" ${index >= VISIBLE ? 'data-extra hidden' : ''}>
        <div class="q-issue-head"><span class="chip sev-${issue.severity}">${escapeHtml(issue.severity)}</span> <strong>${escapeHtml(issue.title)}</strong></div>
        <p>${escapeHtml(issue.detail)}</p>
        <p class="q-why">${escapeHtml(issue.why || '')}</p>
        ${examples}
        ${fix ? `<button type="button" class="btn secondary card-small-btn" data-fix-index="${index}">${escapeHtml(FIX_LABELS[fix.type] || 'Fix')}</button>` : ''}
      </div>`;
  }).join('') + (issues.length > VISIBLE ? `<button type="button" class="insights-toggle" id="quality-toggle">+${issues.length - VISIBLE} more</button>` : '');
  host.querySelectorAll('[data-fix-index]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const issue = issues[Number(btn.dataset.fixIndex)];
      if (!issue?.fix || !state.dataset?.dataset_id) return;
      btn.disabled = true;
      btn.textContent = 'Applying...';
      const outcome = await performTransform(issue.fix.type, issue.fix.params || {});
      if (!outcome.ok) {
        btn.disabled = false;
        btn.textContent = 'Failed — retry';
        btn.title = outcome.message;
      }
    });
  });
  const toggle = document.getElementById('quality-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      const hiddenNow = host.querySelector('[data-extra]').hidden;
      host.querySelectorAll('[data-extra]').forEach((el) => { el.hidden = !hiddenNow; });
      toggle.textContent = hiddenNow ? 'Show less' : `+${issues.length - VISIBLE} more`;
    });
  }
}

// ---- Export page: every export has a file name and its own customization panel. Choices are
// remembered in this browser (localStorage) and sent to the server as an `options` object, which
// validates them again - the browser is never trusted to send sane values.

const EXPORT_PREFS_KEY = 'datascopia.exportPrefs.v2';
const EXPORT_DEFAULTS = {
  notebook: {
    filename: '', title: '', author: '', notes: '',
    sections: {quality: true, cleaning: true, structure: true, summary: true, visualizations: true, statistics: true, modelling: true, findings: true, conclusions: true},
    embed_rows: 10000, chart_source: 'both', max_charts: 12,
    ml: {test_size: 0.25, seed: 42, trees: 200},
  },
  pdf: {
    filename: '', title: '', subtitle: '', author: '', organization: '', notes: '', footer: '',
    detail: 'deep', theme: 'indigo', page_size: 'letter', font_size: 'normal',
    sections: {introduction: true, methods: true, quality: true, results: true, discussion: true, conclusions: true, references: true, appendices: true},
    include_charts: true, include_hypotheses: true, include_evidence: true, include_figures_native: true,
  },
  quality: {
    filename: '', format: 'md', title: '', author: '', notes: '',
    severities: {high: true, medium: true, low: true}, include_why: true, include_examples: true, include_fixes: true, group_by: 'category',
  },
};

function deepMerge(base, extra) {
  const out = Array.isArray(base) ? [...base] : {...base};
  if (!extra || typeof extra !== 'object') return out;
  Object.keys(base).forEach((key) => {
    if (!(key in extra)) return;
    if (base[key] && typeof base[key] === 'object' && !Array.isArray(base[key])) out[key] = deepMerge(base[key], extra[key]);
    else if (typeof extra[key] === typeof base[key]) out[key] = extra[key];
  });
  return out;
}

function loadExportPrefs() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(EXPORT_PREFS_KEY) || '{}') || {}; } catch (error) { saved = {}; }
  const prefs = {};
  Object.keys(EXPORT_DEFAULTS).forEach((form) => { prefs[form] = deepMerge(EXPORT_DEFAULTS[form], saved[form]); });
  return prefs;
}
const exportPrefs = loadExportPrefs();
function saveExportPrefs() {
  try { localStorage.setItem(EXPORT_PREFS_KEY, JSON.stringify(exportPrefs)); } catch (error) { /* private browsing */ }
}

function getPath(obj, path) { return path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj); }
function setPath(obj, path, value) {
  const keys = path.split('.');
  const last = keys.pop();
  const target = keys.reduce((o, k) => o[k], obj);
  target[last] = value;
}

const THEME_SWATCHES = [['indigo', '#4338ca'], ['teal', '#0f766e'], ['slate', '#334155'], ['crimson', '#be123c'], ['forest', '#15803d']];
const EXPORT_FORMS = [
  {
    id: 'notebook', title: 'Jupyter notebook', button: 'export-notebook-btn', action: 'Export notebook',
    blurb: 'A runnable notebook: data loading, quality audit, your exact cleaning steps, charts, statistical tests and a baseline model.',
    ext: () => '.ipynb', defaultName: (n) => `${n}_analysis`,
    fields: [
      {kind: 'text', path: 'title', label: 'Notebook title', placeholder: (n) => `${n} analysis`},
      {kind: 'text', path: 'author', label: 'Author', placeholder: 'Your name (optional)'},
      {kind: 'textarea', path: 'notes', label: 'Note for readers', placeholder: 'Context shown under the title, e.g. the question being explored'},
      {kind: 'checks', path: 'sections', label: 'Include sections', options: [['quality', 'Quality audit'], ['cleaning', 'Cleaning steps'], ['structure', 'Column structure'], ['summary', 'Summary statistics'],
        ['visualizations', 'Visualizations'], ['statistics', 'Statistical tests'], ['modelling', 'Predictive model'], ['findings', 'Saved findings'], ['conclusions', 'Conclusions']]},
      {kind: 'select', path: 'embed_rows', label: 'Embedded data snapshot', number: true, options: [[10000, 'Up to 10,000 rows (runs anywhere)'], [5000, 'Up to 5,000 rows'], [1000, 'Up to 1,000 rows'], [0, 'None - notebook needs the data file']]},
      {kind: 'select', path: 'chart_source', label: 'Charts to include', options: [['both', 'Workspace + recommended'], ['workspace', 'Workspace charts only'], ['recommended', 'Recommended charts only']]},
      {kind: 'number', path: 'max_charts', label: 'Maximum charts', min: 1, max: 30, step: 1},
      {kind: 'group', label: 'Predictive model settings', fields: [
        {kind: 'select', path: 'ml.test_size', label: 'Held-out share', number: true, options: [[0.2, '20%'], [0.25, '25%'], [0.3, '30%']]},
        {kind: 'number', path: 'ml.trees', label: 'Trees', min: 20, max: 600, step: 10},
        {kind: 'number', path: 'ml.seed', label: 'Random seed', min: 0, max: 99999, step: 1},
      ]},
    ],
    summary: (p) => `${Object.values(p.sections).filter(Boolean).length}/9 sections · ${p.embed_rows ? `${p.embed_rows.toLocaleString()}-row snapshot` : 'no embedded data'} · ${p.max_charts} charts max`,
  },
  {
    id: 'pdf', title: 'PDF research report', button: 'export-report-btn', action: 'Export report (PDF)',
    blurb: 'A shareable report: executive summary, methods, deep results and interpretation, evidence ratings, references.',
    ext: () => '.pdf', defaultName: (n) => `${n}_report`,
    fields: [
      {kind: 'text', path: 'title', label: 'Report title', placeholder: (n) => n},
      {kind: 'text', path: 'subtitle', label: 'Subtitle', placeholder: 'e.g. Quarterly review, wave 2'},
      {kind: 'text', path: 'author', label: 'Author', placeholder: 'Your name'},
      {kind: 'text', path: 'organization', label: 'Organization', placeholder: 'Team, school or company'},
      {kind: 'textarea', path: 'notes', label: "Analyst's note (shown in the introduction)", placeholder: 'Why this analysis was run, who it is for, or caveats you want readers to see first'},
      {kind: 'segmented', path: 'detail', label: 'Level of detail', options: [['brief', 'Brief'], ['standard', 'Standard'], ['deep', 'Deep']],
        hint: 'Brief: key tables and conclusions. Standard: adds robustness checks. Deep: adds post-hoc tests, trends, diagnostics and per-chart analysis.'},
      {kind: 'swatches', path: 'theme', label: 'Colour theme', options: THEME_SWATCHES},
      {kind: 'segmented', path: 'page_size', label: 'Page size', options: [['letter', 'US Letter'], ['a4', 'A4']]},
      {kind: 'segmented', path: 'font_size', label: 'Text size', options: [['compact', 'Compact'], ['normal', 'Normal'], ['large', 'Large']]},
      {kind: 'checks', path: 'sections', label: 'Include sections', options: [['introduction', 'Introduction'], ['methods', 'Methods'], ['quality', 'Data quality'], ['results', 'Results'],
        ['discussion', 'Discussion'], ['conclusions', 'Conclusions'], ['references', 'References'], ['appendices', 'Appendices']]},
      {kind: 'toggle', path: 'include_charts', label: 'Include my workspace charts (with analysis)'},
      {kind: 'toggle', path: 'include_figures_native', label: 'Include built-in figures (distributions, missing data, correlation matrix)'},
      {kind: 'toggle', path: 'include_hypotheses', label: 'State testable hypotheses'},
      {kind: 'toggle', path: 'include_evidence', label: 'Include the evidence-rating table'},
      {kind: 'text', path: 'footer', label: 'Footer text', placeholder: 'e.g. Confidential - internal use'},
    ],
    summary: (p) => `${p.detail} detail · ${p.theme} · ${p.page_size === 'a4' ? 'A4' : 'Letter'} · ${Object.values(p.sections).filter(Boolean).length}/8 sections`,
  },
  {
    id: 'quality', title: 'Data quality report', button: 'export-quality-btn', action: 'Download report',
    blurb: 'The scored audit and its suggested fixes, in the format you need.',
    ext: (p) => `.${p.format}`, defaultName: (n) => `${n}_data_quality`,
    fields: [
      {kind: 'segmented', path: 'format', label: 'Format', options: [['md', 'Markdown'], ['json', 'JSON'], ['csv', 'CSV']]},
      {kind: 'text', path: 'title', label: 'Title', placeholder: (n) => `Data quality report: ${n}`},
      {kind: 'text', path: 'author', label: 'Author', placeholder: 'Your name (optional)'},
      {kind: 'textarea', path: 'notes', label: 'Note', placeholder: 'Context for whoever receives this report'},
      {kind: 'checks', path: 'severities', label: 'Severities to include', options: [['high', 'High'], ['medium', 'Medium'], ['low', 'Low']]},
      {kind: 'toggle', path: 'include_why', label: 'Explain why each issue matters'},
      {kind: 'toggle', path: 'include_examples', label: 'Show example values'},
      {kind: 'toggle', path: 'include_fixes', label: 'Show suggested fixes'},
      {kind: 'segmented', path: 'group_by', label: 'Group issues by', options: [['category', 'Category'], ['severity', 'Severity']]},
    ],
    summary: (p) => `${p.format.toUpperCase()} · ${Object.entries(p.severities).filter(([, v]) => v).map(([k]) => k).join(', ') || 'no severities'}`,
  },
];

function datasetLabel() {
  return (state.analysis?.dataset_name || 'dataset').replace(/[^A-Za-z0-9_-]+/g, '_');
}

function fieldHtml(form, field) {
  const prefs = exportPrefs[form.id];
  const name = datasetLabel();
  const value = field.path ? getPath(prefs, field.path) : undefined;
  const ph = typeof field.placeholder === 'function' ? field.placeholder(name) : (field.placeholder || '');
  const path = field.path ? `data-form="${form.id}" data-path="${field.path}"` : '';
  switch (field.kind) {
    case 'text':
      return `<label class="ef"><span>${escapeHtml(field.label)}</span><input type="text" ${path} value="${escapeHtml(value || '')}" placeholder="${escapeHtml(ph)}" maxlength="140" /></label>`;
    case 'textarea':
      return `<label class="ef ef-wide"><span>${escapeHtml(field.label)}</span><textarea ${path} rows="2" maxlength="1500" placeholder="${escapeHtml(ph)}">${escapeHtml(value || '')}</textarea></label>`;
    case 'number':
      return `<label class="ef"><span>${escapeHtml(field.label)}</span><input type="number" ${path} value="${escapeHtml(value)}" min="${field.min}" max="${field.max}" step="${field.step}" /></label>`;
    case 'select':
      return `<label class="ef"><span>${escapeHtml(field.label)}</span><select ${path} ${field.number ? 'data-number="1"' : ''}>${field.options.map(([v, l]) => `<option value="${escapeHtml(v)}" ${String(v) === String(value) ? 'selected' : ''}>${escapeHtml(l)}</option>`).join('')}</select></label>`;
    case 'checks':
      return `<fieldset class="ef ef-wide ef-checks"><legend>${escapeHtml(field.label)}</legend>${field.options.map(([k, l]) => `<label class="ef-check"><input type="checkbox" data-form="${form.id}" data-path="${field.path}.${k}" ${value?.[k] ? 'checked' : ''} /> ${escapeHtml(l)}</label>`).join('')}</fieldset>`;
    case 'toggle':
      return `<label class="ef ef-wide ef-toggle"><input type="checkbox" ${path} ${value ? 'checked' : ''} /> <span>${escapeHtml(field.label)}</span></label>`;
    case 'segmented':
      return `<div class="ef ef-wide"><span class="ef-label">${escapeHtml(field.label)}</span><div class="seg" role="radiogroup" data-form="${form.id}" data-seg="${field.path}">${field.options.map(([v, l]) => `<button type="button" role="radio" aria-checked="${v === value}" class="${v === value ? 'on' : ''}" data-value="${escapeHtml(v)}">${escapeHtml(l)}</button>`).join('')}</div>${field.hint ? `<small class="ef-hint">${escapeHtml(field.hint)}</small>` : ''}</div>`;
    case 'swatches':
      return `<div class="ef ef-wide"><span class="ef-label">${escapeHtml(field.label)}</span><div class="seg swatches" role="radiogroup" data-form="${form.id}" data-seg="${field.path}">${field.options.map(([v, c]) => `<button type="button" role="radio" aria-checked="${v === value}" title="${escapeHtml(v)}" class="${v === value ? 'on' : ''}" data-value="${escapeHtml(v)}" style="--sw:${c}"><i></i>${escapeHtml(v)}</button>`).join('')}</div></div>`;
    case 'group':
      return `<fieldset class="ef ef-wide ef-group"><legend>${escapeHtml(field.label)}</legend><div class="ef-grid">${field.fields.map((f) => fieldHtml(form, f)).join('')}</div></fieldset>`;
    default:
      return '';
  }
}

function renderExportForms() {
  const grid = document.getElementById('export-grid');
  if (!grid) return;
  const name = datasetLabel();
  grid.innerHTML = EXPORT_FORMS.map((form) => {
    const prefs = exportPrefs[form.id];
    return `
      <div class="panel export-card" data-export="${form.id}">
        <div class="export-card-head">
          <h3>${escapeHtml(form.title)}</h3>
          <button type="button" class="ef-reset" data-reset="${form.id}" title="Restore the default settings">Reset</button>
        </div>
        <p class="hint">${escapeHtml(form.blurb)}</p>
        <label class="ef ef-filename"><span>File name</span>
          <span class="ef-file"><input type="text" data-form="${form.id}" data-path="filename" value="${escapeHtml(prefs.filename || '')}" placeholder="${escapeHtml(form.defaultName(name))}" maxlength="90" /><b class="ext" data-ext="${form.id}">${escapeHtml(form.ext(prefs))}</b></span>
        </label>
        <details class="ef-details" ${prefs.__open ? 'open' : ''} data-details="${form.id}">
          <summary>Customize<span class="ef-summary" data-summary="${form.id}">${escapeHtml(form.summary(prefs))}</span></summary>
          <div class="ef-grid">${form.fields.map((f) => fieldHtml(form, f)).join('')}</div>
        </details>
        <button type="button" class="btn" id="${form.button}" data-export-action="${form.id}">${escapeHtml(form.action)}</button>
      </div>`;
  }).join('');
}

function refreshExportSummaries() {
  EXPORT_FORMS.forEach((form) => {
    const summary = document.querySelector(`[data-summary="${form.id}"]`);
    if (summary) summary.textContent = form.summary(exportPrefs[form.id]);
    const ext = document.querySelector(`[data-ext="${form.id}"]`);
    if (ext) ext.textContent = form.ext(exportPrefs[form.id]);
  });
}

(function bindExportForms() {
  const grid = document.getElementById('export-grid');
  if (!grid) return;
  const apply = (el) => {
    const form = el.dataset.form;
    const path = el.dataset.path;
    if (!form || !path) return;
    let value;
    if (el.type === 'checkbox') value = el.checked;
    else if (el.type === 'number' || el.dataset.number) {
      value = Number(el.value);
      if (Number.isNaN(value)) return;
    } else value = el.value;
    setPath(exportPrefs[form], path, value);
    saveExportPrefs();
    refreshExportSummaries();
  };
  grid.addEventListener('input', (event) => { if (event.target.matches('input[type=text], textarea, input[type=number]')) apply(event.target); });
  grid.addEventListener('change', (event) => { if (event.target.matches('input[type=checkbox], select')) apply(event.target); });
  grid.addEventListener('click', (event) => {
    const seg = event.target.closest('.seg button');
    if (seg) {
      const holder = seg.closest('.seg');
      setPath(exportPrefs[holder.dataset.form], holder.dataset.seg, seg.dataset.value);
      holder.querySelectorAll('button').forEach((b) => { b.classList.toggle('on', b === seg); b.setAttribute('aria-checked', String(b === seg)); });
      saveExportPrefs();
      refreshExportSummaries();
      return;
    }
    const reset = event.target.closest('[data-reset]');
    if (reset) {
      exportPrefs[reset.dataset.reset] = JSON.parse(JSON.stringify(EXPORT_DEFAULTS[reset.dataset.reset]));
      saveExportPrefs();
      renderExportForms();
      return;
    }
    const action = event.target.closest('[data-export-action]');
    if (action) startExport(action.dataset.exportAction);
  });
  grid.addEventListener('toggle', (event) => {
    const details = event.target.closest?.('[data-details]');
    if (details) { exportPrefs[details.dataset.details].__open = details.open; }
  }, true);
})();

// Every export asks for a title (and confirms the file name) first, so nothing is generated
// under an accidental default name. The answer is saved so the next export starts from it.
const EXPORT_TITLE_COPY = {
  notebook: {heading: 'Title your notebook', hint: 'This becomes the notebook\'s heading and its file name.', confirm: 'Export notebook', fallback: (n) => `${n} analysis`},
  pdf: {heading: 'Title your report', hint: 'This appears on the report\'s title block and in the page headers.', confirm: 'Export report', fallback: (n) => `${n}: data analysis report`},
  quality: {heading: 'Title your data quality report', hint: 'This becomes the report heading and its file name.', confirm: 'Download report', fallback: (n) => `Data quality report: ${n}`},
};

function fileNameFromTitle(title) {
  return title.trim().replace(/[^A-Za-z0-9 _.-]+/g, '').replace(/\s+/g, '_').replace(/^[._]+|[._]+$/g, '').slice(0, 90);
}

function promptExportDetails(id) {
  const copy = EXPORT_TITLE_COPY[id];
  const form = EXPORT_FORMS.find((f) => f.id === id);
  const prefs = exportPrefs[id];
  const overlay = document.getElementById('export-title-overlay');
  const titleInput = document.getElementById('export-title-input');
  const fileInput = document.getElementById('export-title-file');
  const errorEl = document.getElementById('export-title-error');
  const confirmBtn = document.getElementById('export-title-confirm');
  const cancelBtn = document.getElementById('export-title-cancel');
  document.getElementById('export-title-heading').textContent = copy.heading;
  document.getElementById('export-title-hint').textContent = copy.hint;
  document.getElementById('export-title-ext').textContent = form.ext(prefs);
  confirmBtn.textContent = copy.confirm;
  // Default the title to whatever the user already called this analysis (the saved/cloud project
  // name), so they aren't asked to type essentially the same name again for every export; falls
  // back to the dataset name only when nothing has been named yet.
  const dataset = state.projectName || state.analysis?.dataset_name || 'dataset';
  titleInput.value = prefs.title || copy.fallback(dataset);
  fileInput.value = prefs.filename || fileNameFromTitle(titleInput.value);
  // Only treat the file name as "manually customized" (and stop following the title) if it actually
  // diverges from what the current title would derive - otherwise a title carried over from a
  // previous export (or a stale prefs.filename) would silently stop the two fields from matching.
  let fileEdited = Boolean(prefs.filename) && prefs.filename !== fileNameFromTitle(titleInput.value);
  errorEl.hidden = true;
  overlay.hidden = false;
  titleInput.focus();
  titleInput.select();

  return new Promise((resolve) => {
    const cleanup = (result) => {
      overlay.hidden = true;
      titleInput.removeEventListener('input', onTitle);
      fileInput.removeEventListener('input', onFile);
      confirmBtn.removeEventListener('click', onConfirm);
      cancelBtn.removeEventListener('click', onCancel);
      overlay.removeEventListener('click', onBackdrop);
      document.removeEventListener('keydown', onKey);
      resolve(result);
    };
    const onTitle = () => {
      errorEl.hidden = true;
      if (!fileEdited) fileInput.value = fileNameFromTitle(titleInput.value);
    };
    const onFile = () => { fileEdited = true; };
    const onConfirm = () => {
      const title = titleInput.value.trim();
      if (!title) { errorEl.hidden = false; titleInput.focus(); return; }
      cleanup({title, filename: fileInput.value.trim() || fileNameFromTitle(title)});
    };
    const onCancel = () => cleanup(null);
    const onBackdrop = (event) => { if (event.target === overlay) cleanup(null); };
    const onKey = (event) => {
      if (event.key === 'Escape') cleanup(null);
      if (event.key === 'Enter' && !overlay.hidden) { event.preventDefault(); onConfirm(); }
    };
    titleInput.addEventListener('input', onTitle);
    fileInput.addEventListener('input', onFile);
    confirmBtn.addEventListener('click', onConfirm);
    cancelBtn.addEventListener('click', onCancel);
    overlay.addEventListener('click', onBackdrop);
    document.addEventListener('keydown', onKey);
  });
}

async function startExport(id) {
  if (!state.analysis) return;
  const details = await promptExportDetails(id);
  if (!details) return;
  exportPrefs[id].title = details.title;
  exportPrefs[id].filename = details.filename;
  saveExportPrefs();
  // keep the card's own fields in step with what was just chosen
  const titleField = document.querySelector(`[data-export="${id}"] input[data-path="title"]`);
  const nameField = document.querySelector(`[data-export="${id}"] input[data-path="filename"]`);
  if (titleField) titleField.value = details.title;
  if (nameField) nameField.value = details.filename;
  if (id === 'notebook') await exportNotebook();
  else if (id === 'pdf') await exportReportPDF();
  else await downloadQualityReport();
}

function exportPayloadOptions(form) {
  const copy = JSON.parse(JSON.stringify(exportPrefs[form]));
  delete copy.filename;
  delete copy.__open;
  return copy;
}

function saveBlobAs(blob, fallbackName, response) {
  const name = response?.headers?.get('X-Filename') || fallbackName;
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
  return name;
}

async function withBusyButton(button, busyLabel, work) {
  const original = button ? button.textContent : '';
  if (button) { button.disabled = true; button.textContent = busyLabel; }
  try {
    return await work();
  } finally {
    if (button) { button.disabled = false; button.textContent = original; }
  }
}

function flashDone(button, message) {
  if (!button) return;
  const original = button.textContent;
  button.textContent = message;
  setTimeout(() => { button.textContent = original; }, 1800);
}

async function downloadQualityReport() {
  if (!state.analysis) return;
  const button = document.getElementById('export-quality-btn');
  const prefs = exportPrefs.quality;
  await withBusyButton(button, 'Preparing...', async () => {
    const response = await fetch(`${API_BASE}/analysis/quality-report`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        analysis: {data_quality: state.analysis.data_quality, dataset_name: state.analysis.dataset_name},
        dataset_name: state.analysis.dataset_name,
        filename: prefs.filename,
        options: {...exportPayloadOptions('quality'), severities: Object.entries(prefs.severities).filter(([, v]) => v).map(([k]) => k)},
      }),
    });
    if (!response.ok) { alert('Could not build the data quality report.'); return; }
    saveBlobAs(await response.blob(), `${datasetLabel()}_data_quality.${prefs.format}`, response);
    flashDone(button, 'Downloaded ✓');
  });
}
document.getElementById('quality-download-btn').addEventListener('click', () => startExport('quality'));

// Reproducible notebook: the server rebuilds it from the cached data plus the exact cleaning
// steps applied here (transformLog), the charts currently on screen and the saved findings, so
// running it top to bottom yields the same cleaned data, charts and statistics.
async function exportNotebook() {
  if (!state.analysis) return;
  const button = document.getElementById('export-notebook-btn');
  const prefs = exportPrefs.notebook;
  await withBusyButton(button, 'Building notebook...', async () => {
    try {
      const response = await fetch(`${API_BASE}/analysis/notebook`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          title: prefs.title.trim() || `${state.analysis.dataset_name || 'Dataset'} analysis`,
          filename: prefs.filename,
          options: exportPayloadOptions('notebook'),
          dataset_id: state.dataset?.dataset_id,
          dataset: state.dataset,
          analysis: state.analysis,
          findings: state.findings,
          charts: chartSlots.filter((slot) => slot.x.value || slot.type.value === 'missing').map((slot) => chartConfigOf(slot)),
          transform_log: state.transformLog.filter((entry) => entry && entry.type),
        }),
      });
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.detail || `Notebook export failed (${response.status}).`);
      }
      const notebook = await response.json();
      saveBlobAs(new Blob([JSON.stringify(notebook, null, 1)], {type: 'application/json'}), `${datasetLabel()}_analysis.ipynb`, response);
      flashDone(button, 'Downloaded ✓');
    } catch (error) {
      alert(`Could not export the notebook: ${error.message || error}`);
    }
  });
}


const TYPE_LABELS = {numeric: 'Numeric', categorical: 'Categorical', datetime: 'Datetime', boolean: 'Boolean'};
// What each detected role means for analysis, shown in the Column Explorer.
const ROLE_HINTS = {
  identifier: 'Labels rows rather than measuring anything, so it is left out of statistics, correlations and models.',
  year: 'A calendar year - used as a time axis, not averaged like a measurement.',
  datetime: 'Dates/times - can drive time-series charts.',
  geo_country: 'Countries - can be drawn on a map and used to join other country data.',
  geo_state: 'US states - can be drawn on a map.',
  geo_lat: 'Latitude - can place points on a map.',
  geo_lon: 'Longitude - can place points on a map.',
  binary: 'Two possible values (yes/no style).',
  ordinal: 'Categories with a natural order (e.g. low / medium / high).',
  ordinal_numeric: 'A small set of ordered numeric levels (e.g. ratings).',
  text: 'Free text - not charted as categories.',
  constant: 'Every value is identical, so it carries no information.',
  empty: 'No values at all.',
};

function renderColumnExplorer() {
  const columns = state.analysis?.columns || [];
  if (!columns.length) {
    columnExplorerSelect.disabled = true;
    columnExplorerSelect.innerHTML = '<option>No dataset loaded yet</option>';
    columnExplorerBody.innerHTML = '<p class="hint">Load a dataset to explore its columns.</p>';
    return;
  }
  const previousValue = columnExplorerSelect.value;
  const columnDescriptions = state.dataset?.column_descriptions || {};
  columnExplorerSelect.disabled = false;
  columnExplorerSelect.innerHTML = columns
    .map((col) => `<option value="${escapeHtml(col.name)}">${escapeHtml(col.name)}${columnDescriptions[col.name] ? ' ✓' : ''}</option>`)
    .join('');
  const stillValid = columns.some((col) => col.name === previousValue);
  columnExplorerSelect.value = stillValid ? previousValue : columns[0].name;
  renderColumnExplorerDetail(columnExplorerSelect.value);
}

function renderColumnExplorerDetail(columnName) {
  const columns = state.analysis?.columns || [];
  const meta = columns.find((col) => col.name === columnName);
  if (!meta) {
    columnExplorerBody.innerHTML = '<p class="hint">Column not found.</p>';
    return;
  }
  const columnDescriptions = state.dataset?.column_descriptions || {};
  const inferredDescriptions = state.dataset?.inferred_column_descriptions || {};
  const documented = columnDescriptions[columnName];
  const inferred = inferredDescriptions[columnName];
  const numericSummary = state.analysis?.numeric_summary?.[columnName];
  const typeLabel = meta.role_label || TYPE_LABELS[meta.type] || meta.type;
  const roleHint = ROLE_HINTS[meta.role] || '';
  const storedAsText = meta.notes?.stored_as_text ? '<p class="ce-inferred-tag">Stored as text; read as a real number/date for analysis.</p>' : '';

  const statCards = [
    `<div class="ce-stat"><span class="label">Missing</span><span class="value">${meta.missing_pct}%</span></div>`,
    `<div class="ce-stat"><span class="label">Unique values</span><span class="value">${meta.unique_count.toLocaleString()}</span></div>`,
  ];
  if (numericSummary) {
    statCards.push(
      `<div class="ce-stat"><span class="label">Mean</span><span class="value">${numericSummary.mean}</span></div>`,
      `<div class="ce-stat"><span class="label">Median</span><span class="value">${numericSummary.median}</span></div>`,
      `<div class="ce-stat"><span class="label">Min</span><span class="value">${numericSummary.min}</span></div>`,
      `<div class="ce-stat"><span class="label">Max</span><span class="value">${numericSummary.max}</span></div>`,
    );
  }

  let descriptionHtml;
  if (documented) {
    descriptionHtml = `<p>${boldify(escapeHtml(documented))}</p>`;
  } else if (inferred) {
    descriptionHtml = `
      <p class="ce-inferred-tag">Approximate — inferred from the column name, not from the dataset's documentation</p>
      <p>${boldify(escapeHtml(inferred))}</p>
    `;
  } else {
    descriptionHtml = `<p class="ce-no-desc">No description could be determined for this column — its name and values are the best guide to what it represents.</p>`;
  }

  columnExplorerBody.innerHTML = `
    <div class="column-explorer-detail">
      <div class="ce-header">
        <span class="ce-name">${escapeHtml(columnName)}</span>
        <span class="type-badge type-${meta.type}">${escapeHtml(typeLabel)}</span>
      </div>
      ${roleHint ? `<p class="ce-role-hint">${escapeHtml(roleHint)}</p>` : ''}${storedAsText}
      <div class="ce-stats">${statCards.join('')}</div>
      <div class="ce-description">
        <h4>What this column represents</h4>
        ${descriptionHtml}
      </div>
    </div>
  `;
}

columnExplorerSelect.addEventListener('change', () => renderColumnExplorerDetail(columnExplorerSelect.value));

function renderFindings() {
  window.markProjectDirty?.();
  findingsEl.hidden = state.findings.length === 0;
  document.getElementById('findings-empty').hidden = state.findings.length > 0;
  const badge = document.getElementById('nav-findings-count');
  badge.hidden = state.findings.length === 0;
  badge.textContent = state.findings.length;
  findingsListEl.innerHTML = state.findings.map((finding) => {
    const detail = finding.type === 'question' ? finding.answer : (finding.summary || '');
    return `<li><strong>${escapeHtml(finding.title)}</strong><div style="color: var(--muted); font-size: 0.85rem;">${escapeHtml(detail)}</div></li>`;
  }).join('');
  renderExportSummary();
}

function renderExportSummary() {
  const el = document.getElementById('export-summary');
  if (!el) return;
  const steps = state.transformLog.filter((entry) => entry && entry.type).length;
  el.textContent = `Your export will include ${chartSlots.length} chart${chartSlots.length === 1 ? '' : 's'} from the Workspace, `
    + `${steps} cleaning step${steps === 1 ? '' : 's'} and ${state.findings.length} saved finding${state.findings.length === 1 ? '' : 's'}.`;
}

// ---- Past analyses: localStorage save/resume (server holds nothing beyond a session) ----

const SAVED_ANALYSES_KEY = 'kaggleWorkspace.savedAnalyses.v1';

function loadSavedAnalyses() {
  try {
    const raw = localStorage.getItem(SAVED_ANALYSES_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    return [];
  }
}

function persistSavedAnalyses(list) {
  try {
    localStorage.setItem(SAVED_ANALYSES_KEY, JSON.stringify(list));
    return true;
  } catch (error) {
    return false;
  }
}

function buildAnalysisSnapshot(name) {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    name,
    savedAt: new Date().toISOString(),
    dataset: state.dataset,
    analysis: state.analysis,
    slug: state.slug,
    findings: state.findings,
    transformLog: state.transformLog,
    chartConfigs: chartSlots.map((slot) => ({
      label: slot.label, ...chartConfigOf(slot),
    })),
  };
}

function saveCurrentAnalysis(name) {
  if (!state.analysis) return {ok: false, message: 'No analysis loaded to save.'};
  const trimmedName = (name || '').trim() || `${state.analysis.dataset_name || 'Analysis'} — ${new Date().toLocaleString()}`;
  const list = loadSavedAnalyses();
  list.unshift(buildAnalysisSnapshot(trimmedName));
  const ok = persistSavedAnalyses(list);
  if (ok) state.projectName = trimmedName;
  return ok ? {ok: true} : {ok: false, message: 'Could not save — your browser storage may be full or unavailable (e.g. private browsing).'};
}

function deleteSavedAnalysis(id) {
  persistSavedAnalyses(loadSavedAnalyses().filter((item) => item.id !== id));
  renderSavedAnalysesPanel();
}

function restoreSavedAnalysis(id) {
  const item = loadSavedAnalyses().find((entry) => entry.id === id);
  if (!item) return;
  state.dataset = item.dataset;
  state.analysis = item.analysis;
  state.slug = item.slug || null;
  state.projectId = null;
  state.projectName = item.name || null;
  state.findings = item.findings || [];
  // older saves stored just the description text; keep those readable (they can't be replayed)
  state.transformLog = (item.transformLog || []).map((entry) => (typeof entry === 'string' ? {description: entry} : entry));
  downloadFullBtn.hidden = !state.slug;
  fullAnalysisBtn.hidden = true;
  resetTransformBtn.hidden = state.transformLog.length === 0;

  const fields = Object.keys((state.analysis?.preview || [])[0] || {});
  if (fields.length) {
    setVizSectionVisible(true);
    rebuildChartSlots(fields, [], item.chartConfigs);
    resetDatasetEditorSlots();
  }
  renderAnalysis({rebuildCharts: false});
  navigate('overview');
}

function renderSavedAnalysesPanel() {
  const list = loadSavedAnalyses();
  if (!list.length) {
    pastAnalysesListEl.innerHTML = '<div class="result-item">No saved analyses yet. Load a dataset, then click &quot;Save analysis&quot;.</div>';
    return;
  }
  pastAnalysesListEl.innerHTML = list.map((item) => `
    <div class="past-analysis-item">
      <strong>${escapeHtml(item.name)}</strong>
      <div class="meta">${escapeHtml(item.dataset?.name || item.analysis?.dataset_name || 'dataset')} · ${escapeHtml(new Date(item.savedAt).toLocaleString())}</div>
      <button type="button" class="btn secondary card-small-btn" style="margin-right: 6px;" data-load-id="${item.id}">Load</button>
      <button type="button" class="btn secondary card-small-btn" data-delete-id="${item.id}">Delete</button>
    </div>
  `).join('');
  pastAnalysesListEl.querySelectorAll('[data-load-id]').forEach((btn) => {
    btn.addEventListener('click', () => restoreSavedAnalysis(btn.dataset.loadId));
  });
  pastAnalysesListEl.querySelectorAll('[data-delete-id]').forEach((btn) => {
    btn.addEventListener('click', () => {
      if (confirm('Delete this saved analysis? This cannot be undone.')) deleteSavedAnalysis(btn.dataset.deleteId);
    });
  });
}

// ---- License visibility: the license is shown in search results and on the loaded dataset
// (no blocking dialog - users are informed, never interrupted).

const LICENSE_LEVELS = {
  open: {label: 'Open', className: 'lic-open'},
  conditions: {label: 'Conditions apply', className: 'lic-conditions'},
  restricted: {label: 'Restricted — read the terms', className: 'lic-restricted'},
  unknown: {label: 'No license listed', className: 'lic-unknown'},
};

function licenseSummaryHtml(license) {
  const level = LICENSE_LEVELS[license?.level] || LICENSE_LEVELS.unknown;
  return `
    <div class="license-name">${escapeHtml(license?.name || 'Not specified')} <span class="chip ${level.className}">${level.label}</span></div>
    <p>${escapeHtml(license?.summary || '')}</p>`;
}

async function analyzeSearchResult(slug) {
  await loadDatasetFromSlug(slug);
}

function renderLicenseLine() {
  const el = document.getElementById('license-line');
  const info = state.dataset?.license_info;
  if (!info) {
    el.hidden = true;
    el.innerHTML = '';
    return;
  }
  el.hidden = false;
  el.innerHTML = `<strong>License</strong> ${licenseSummaryHtml(info)}`;
}

async function searchDatasets() {
  const query = searchInput.value.trim();
  if (!query) {
    resultsCountEl.textContent = '';
    resultsEl.innerHTML = '<div class="result-item">Enter a dataset keyword to search.</div>';
    return;
  }
  resultsCountEl.textContent = '';
  resultsEl.innerHTML = '<div class="result-item">Searching Kaggle...</div>';
  try {
    const response = await fetch(`${API_BASE}/datasets/search?query=${encodeURIComponent(query)}&limit=150`);
    const data = await response.json();
    const results = data.results || [];
    if (!results.length) {
      resultsEl.innerHTML = '<div class="result-item">No results found.</div>';
      return;
    }

    resultsCountEl.textContent = `${results.length.toLocaleString()} dataset${results.length === 1 ? '' : 's'} found — scroll for more`;
    resultsEl.innerHTML = results
      .map((item) => `
        <div class="result-item">
          <strong>${escapeHtml(item.title || 'Dataset')}</strong>
          <div>${escapeHtml(item.subtitle || '')}</div>
          <div style="margin-top: 8px; color: #9ca3af;">${escapeHtml(item.size || '')}${item.is_demo ? '' : `<span class="chip license-chip">License: ${escapeHtml(item.license || 'check on Kaggle')}</span>`}</div>
          <button class="btn secondary" style="margin-top: 10px;" data-analyze-slug="${escapeHtml(item.slug || item.id || 'demo')}">Analyze</button>
          ${item.url ? `<a class="btn secondary" style="display: inline-block; margin: 10px 0 0 6px; text-decoration: none;" href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">View on Kaggle</a>` : ''}
        </div>
      `)
      .join('');
    resultsEl.querySelectorAll('[data-analyze-slug]').forEach((btn) => {
      btn.addEventListener('click', () => analyzeSearchResult(btn.dataset.analyzeSlug));
    });
  } catch (error) {
    resultsCountEl.textContent = '';
    resultsEl.innerHTML = '<div class="result-item">Search failed. Start the API server first.</div>';
  }
}

async function uploadDataset() {
  const file = uploadInput.files[0];
  if (!file) {
    resultsEl.innerHTML = '<div class="result-item">Choose a file first.</div>';
    return;
  }
  const formData = new FormData();
  formData.append('file', file);
  uploadBtn.textContent = 'Analyzing...';
  uploadBtn.disabled = true;
  try {
    const response = await fetch(`${API_BASE}/datasets/upload`, {method: 'POST', body: formData});
    const json = await response.json();
    if (!response.ok) {
      resultsEl.innerHTML = `<div class="result-item"><strong>Upload failed</strong><div>${json.detail || 'Could not parse this file.'}</div></div>`;
      return;
    }
    state.dataset = json.dataset;
    state.analysis = json.analysis;
    state.slug = null;
    state.transformLog = [];
    state.projectId = null;
    state.projectName = null;
    resetTransformBtn.hidden = true;
    downloadFullBtn.hidden = true;
    fullAnalysisBtn.hidden = true;
    renderAnalysis();
    navigate('overview');
    startOnboarding();
  } catch (error) {
    resultsEl.innerHTML = '<div class="result-item">Upload failed. Start the API server first.</div>';
  } finally {
    uploadBtn.textContent = 'Upload';
    uploadBtn.disabled = false;
  }
}

async function loadDatasetFromSlug(slug, endpoint = '/datasets/analyze') {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ slug }),
  });

  if (!response.ok) {
    const payload = await response.json();
    resultsEl.innerHTML = `<div class="result-item"><strong>Dataset unavailable</strong><div>${payload.detail || 'Unable to load this dataset from Kaggle.'}</div><div style="margin-top: 8px; color: #9ca3af;">This search result may have been removed or may require Kaggle access.</div></div>`;
    return;
  }

  const json = await response.json();
  state.dataset = json.dataset;
  state.analysis = json.analysis;
  state.slug = slug;
  state.transformLog = [];
  state.projectId = null;
  state.projectName = null;
  resetTransformBtn.hidden = true;
  downloadFullBtn.hidden = false;
  fullAnalysisBtn.hidden = !(state.analysis?.sample?.is_sample);
  renderAnalysis();
  navigate('overview');
  startOnboarding();
}

async function loadDemoData() {
  const response = await fetch(`${API_BASE}/datasets/demo`);
  const json = await response.json();
  state.dataset = json.dataset;
  state.analysis = json.analysis;
  state.slug = null;
  state.transformLog = [];
  state.projectId = null;
  state.projectName = null;
  resetTransformBtn.hidden = true;
  downloadFullBtn.hidden = true;
  fullAnalysisBtn.hidden = true;
  renderAnalysis();
  navigate('overview');
  startOnboarding();
}

function renderAnalysis(options = {}) {
  const {rebuildCharts = true} = options;
  if (!state.analysis) return;
  saveAnalysisBtn.hidden = false;
  const pill = document.getElementById('nav-dataset');
  pill.hidden = false;
  pill.textContent = state.analysis.dataset_name || 'dataset';
  document.getElementById('site-nav').classList.add('has-data');
  document.getElementById('tour-btn').hidden = false;

  const snap = state.analysis.research_snapshot || {};
  datasetInfoEl.innerHTML = `
    <div><strong>Dataset:</strong> ${state.analysis.dataset_name}</div>
    <div><strong>Rows:</strong> ${snap.rows ?? state.analysis.shape.rows}</div>
    <div><strong>Columns:</strong> ${snap.columns ?? state.analysis.shape.columns}</div>
  `;

  const kaggleInfo = state.dataset?.kaggle_info;
  if (kaggleInfo && kaggleInfo.description) {
    aboutDatasetEl.hidden = false;
    const licenseLine = kaggleInfo.license ? `<p class="hint" style="margin-top: 10px;">License: ${escapeHtml(kaggleInfo.license)}</p>` : '';
    aboutDatasetBodyEl.innerHTML = formatDescription(kaggleInfo.description) + licenseLine;
  } else {
    aboutDatasetEl.hidden = true;
    aboutDatasetBodyEl.innerHTML = '';
  }
  renderColumnExplorer();

  const sample = state.analysis.sample;
  if (sample && sample.is_sample) {
    const sampleMb = sample.sample_bytes
      ? (sample.sample_bytes < 1024 * 1024
          ? `${Math.round(sample.sample_bytes / 1024)} KB`
          : `${(sample.sample_bytes / (1024 * 1024)).toFixed(1)} MB`)
      : 'a partial file';
    const estimate = sample.estimated_total_rows ? ` (~${sample.estimated_total_rows.toLocaleString()} rows estimated total)` : '';
    sampleBannerEl.hidden = false;
    sampleBannerEl.innerHTML = `<div class="result-item">This analysis was computed from a ${sampleMb} streamed sample of <strong>${sample.file_name}</strong>${estimate} — no full download was required. Use "Run full analysis" for complete precision.</div>`;
  } else {
    sampleBannerEl.hidden = true;
    sampleBannerEl.innerHTML = '';
  }

  statsContainer.innerHTML = `
    <div class="stat-card"><span class="label">Rows</span><span class="value">${snap.rows ?? 0}</span></div>
    <div class="stat-card"><span class="label">Columns</span><span class="value">${snap.columns ?? 0}</span></div>
    <div class="stat-card"><span class="label">Missing</span><span class="value">${snap.missing_percentage ?? 0}%</span></div>
    <div class="stat-card"><span class="label">Targets</span><span class="value">${snap.potential_targets ?? 0}</span></div>
    <div class="stat-card"><span class="label">Correlations</span><span class="value">${snap.notable_relationships ?? 0}</span></div>
    <div class="stat-card"><span class="label">Outliers</span><span class="value">${state.analysis.outlier_count ?? 0}</span></div>
    <div class="stat-card"><span class="label">Quality issues</span><span class="value">${snap.data_quality_issues ?? 0}</span></div>
    <div class="stat-card"><span class="label">High-cardinality</span><span class="value">${snap.high_cardinality_columns ?? 0}</span></div>
    <div class="stat-card"><span class="label">Charts suggested</span><span class="value">${snap.recommended_visualizations ?? 0}</span></div>
  `;

  renderCappedList(insightsList, state.analysis.insights, 4, 'No notable insights detected.');
  renderCappedList(questionsList, state.analysis.research_questions, 4, 'No research questions suggested yet.');
  renderQualityPanel();
  renderLicenseLine();

  const fields = Object.keys((state.analysis.preview || [])[0] || {});
  if (fields.length) {
    setVizSectionVisible(true);
    renderRecommendations();
    if (rebuildCharts) {
      state.qa = [];
      renderQaHistory();
      questionAnswer.innerHTML = '';
      rebuildChartSlots(fields, state.analysis.recommendations || []);
      resetDatasetEditorSlots();
      renderTransformLog();
    } else {
      chartSlots.forEach((slot) => {
        refreshChartFieldOptions(slot, fields);
        renderChart(slot);
      });
    }
  } else if (rebuildCharts) {
    setVizSectionVisible(false);
  }
  renderFindings();
}

addChartBtn.addEventListener('click', () => {
  const fields = Object.keys((state.analysis?.preview || [])[0] || {});
  if (!fields.length) return;
  createChartSlot(`Custom chart ${chartSlots.length + 1}`, fields);
  resizeAllCharts();
});
addDatasetEditorBtn.addEventListener('click', () => {
  createDatasetEditorSlot(`Dataset editor ${datasetEditorSlots.length + 1}`);
  resizeAllCharts();
});
resetTransformBtn.addEventListener('click', resetAllTransforms);

window.addEventListener('resize', resizeAllCharts);
document.getElementById('search-btn').addEventListener('click', searchDatasets);
uploadBtn.addEventListener('click', uploadDataset);
fullAnalysisBtn.addEventListener('click', async () => {
  if (!state.slug) return;
  fullAnalysisBtn.textContent = 'Running full analysis...';
  fullAnalysisBtn.disabled = true;
  try {
    await loadDatasetFromSlug(state.slug, '/datasets/analyze-full');
  } finally {
    fullAnalysisBtn.textContent = 'Run full analysis';
    fullAnalysisBtn.disabled = false;
    fullAnalysisBtn.hidden = true;
  }
});
downloadFullBtn.addEventListener('click', async () => {
  if (!state.slug) return;
  const response = await fetch(`${API_BASE}/datasets/download-file`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({slug: state.slug}),
  });
  if (!response.ok) {
    resultsEl.innerHTML = '<div class="result-item">The full dataset download failed. Analysis results are still available.</div>';
    return;
  }
  const blob = await response.blob();
  const link = document.createElement('a');
  link.href = URL.createObjectURL(blob);
  link.download = `${state.slug.replace('/', '-')}.zip`;
  link.click();
  URL.revokeObjectURL(link.href);
});
document.getElementById('load-demo-btn').addEventListener('click', loadDemoData);

searchInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') searchDatasets();
});

saveAnalysisBtn.addEventListener('click', () => {
  if (window.handleSaveClick && window.handleSaveClick()) return;
  saveNameInput.value = state.analysis?.dataset_name ? `${state.analysis.dataset_name} analysis` : '';
  saveNameOverlay.hidden = false;
  saveNameInput.focus();
});
document.getElementById('save-name-cancel-btn').addEventListener('click', () => { saveNameOverlay.hidden = true; });
document.getElementById('save-name-confirm-btn').addEventListener('click', () => {
  if (window.cloudSaveConfirm && window.cloudSaveConfirm(saveNameInput.value)) { saveNameOverlay.hidden = true; return; }
  const result = saveCurrentAnalysis(saveNameInput.value);
  saveNameOverlay.hidden = true;
  if (!result.ok) {
    alert(result.message);
    return;
  }
  if (!pastAnalysesPanel.hidden) renderSavedAnalysesPanel();
});
saveNameInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') document.getElementById('save-name-confirm-btn').click();
  if (event.key === 'Escape') saveNameOverlay.hidden = true;
});
saveNameOverlay.addEventListener('click', (event) => {
  if (event.target === saveNameOverlay) saveNameOverlay.hidden = true;
});
// The on-screen charts are styled for a dark page (light text, transparent background), which is
// unreadable on white paper. For the PDF each chart is redrawn off-screen with a print theme.
function printableLayout(layout) {
  const ink = '#1a2233';
  const muted = '#3b4358';
  const copy = JSON.parse(JSON.stringify(layout || {}));
  copy.paper_bgcolor = '#ffffff';
  copy.plot_bgcolor = '#ffffff';
  copy.font = {...(copy.font || {}), color: ink};
  if (copy.title && typeof copy.title === 'object') copy.title.font = {...(copy.title.font || {}), color: ink};
  if (copy.legend) copy.legend.font = {...(copy.legend.font || {}), color: muted};
  Object.keys(copy).forEach((key) => {
    if (/^[xy]axis\d*$/.test(key) && copy[key] && typeof copy[key] === 'object') {
      const axis = copy[key];
      axis.gridcolor = '#e3e7ef';
      axis.linecolor = '#b8c0d0';
      axis.zerolinecolor = '#b8c0d0';
      axis.tickfont = {...(axis.tickfont || {}), color: muted};
      if (axis.title && typeof axis.title === 'object') axis.title.font = {...(axis.title.font || {}), color: muted};
    }
  });
  if (copy.geo) {
    copy.geo = {...copy.geo, bgcolor: '#ffffff', landcolor: '#eceff5', countrycolor: '#c3cadb', coastlinecolor: '#c3cadb'};
  }
  (copy.annotations || []).forEach((a) => { a.font = {...(a.font || {}), color: muted}; });
  return copy;
}

async function renderPrintableChart(slot) {
  const holder = document.createElement('div');
  holder.style.cssText = 'position: fixed; left: -10000px; top: 0; width: 820px; height: 440px;';
  document.body.appendChild(holder);
  try {
    const data = JSON.parse(JSON.stringify(slot.area.data || []));
    await Plotly.newPlot(holder, data, {...printableLayout(slot.area.layout), width: 820, height: 440}, {staticPlot: true});
    return await Plotly.toImage(holder, {format: 'png', width: 820, height: 440, scale: 2});
  } finally {
    try { Plotly.purge(holder); } catch (error) { /* already gone */ }
    holder.remove();
  }
}

// Shareable report: a server-generated PDF whose numbers come from the full cached dataframe. The
// browser's job is to capture print-themed chart images (only if the user wants them), and pass the
// chosen file name and customization options; the server does the analysis and the writing.
async function exportReportPDF() {
  if (!state.analysis) return;
  const button = document.getElementById('export-report-btn');
  const prefs = exportPrefs.pdf;
  await withBusyButton(button, 'Generating report...', async () => {
    try {
      const charts = [];
      for (const slot of (prefs.include_charts ? chartSlots : [])) {
        if (!slot.area || !slot.area.data || !slot.x || !slot.x.value) continue;
        const title = (slot.titleInput?.value || '').trim() || slot.serverTitle || `${slot.x.value}${slot.y?.value ? ' vs ' + slot.y.value : ''}`;
        let imageBase64 = null;
        try {
          const imgData = await renderPrintableChart(slot);
          imageBase64 = imgData.split(',')[1] || null;
        } catch (error) {
          // A chart that fails to rasterize just loses its image; the analysis still runs.
        }
        charts.push({...chartConfigOf(slot), title, image_base64: imageBase64});
      }
      const response = await fetch(`${API_BASE}/analysis/report`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          dataset_id: state.dataset?.dataset_id,
          dataset_name: state.analysis.dataset_name || 'Dataset',
          filename: prefs.filename,
          options: exportPayloadOptions('pdf'),
          analysis: state.analysis,
          dataset: state.dataset,
          transform_log: state.transformLog.filter((entry) => entry && entry.type),
          charts,
          findings: state.findings,
        }),
      });
      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}));
        throw new Error(errBody.detail || `Report generation failed (${response.status}).`);
      }
      saveBlobAs(await response.blob(), `${datasetLabel()}_report.pdf`, response);
      flashDone(button, 'Downloaded ✓');
    } catch (error) {
      alert(`Could not generate the PDF report: ${error.message || error}`);
      console.error(error);
    }
  });
}
let lastQuestionResult = null;
state.qa = [];

async function runQuestion(question) {
  question = (question || '').trim();
  if (!state.analysis || !question) return;
  questionInput.value = question;
  questionAnswer.innerHTML = '<p>Computing...</p>';
  const response = await fetch(`${API_BASE}/analysis/question`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question, analysis: state.analysis, dataset_id: state.dataset?.dataset_id}),
  });
  const result = await response.json();
  if (!response.ok) {
    lastQuestionResult = null;
    questionAnswer.innerHTML = `<p>${escapeHtml(result.detail || 'Could not answer this question.')}</p>`;
    return;
  }
  lastQuestionResult = {...result, question};
  const badge = result.computed
    ? `<span class="chip" style="background: rgba(52, 211, 153, 0.16); border-color: rgba(52, 211, 153, 0.4);">computed: ${escapeHtml(result.method)}</span>`
    : `<span class="chip" style="background: rgba(248, 113, 113, 0.14); border-color: rgba(248, 113, 113, 0.35);">not computed</span>`;
  const codeLine = result.code ? `<div style="margin-top: 6px; font-family: monospace; font-size: 0.8rem; color: var(--muted);">${escapeHtml(result.code)}</div>` : '';
  const saveBtn = result.computed ? `<button type="button" class="btn secondary" id="save-answer-btn" style="margin-top: 8px; padding: 4px 10px; font-size: 0.78rem;">Save as finding</button>` : '';
  questionAnswer.innerHTML = `<p><strong>Answer</strong> ${badge}</p><p>${escapeHtml(result.answer || 'No answer available.')}</p>${codeLine}${saveBtn}`;
  state.qa.unshift({question, answer: result.answer || '', computed: result.computed});
  renderQaHistory();
  document.getElementById('save-answer-btn')?.addEventListener('click', () => {
    if (!lastQuestionResult) return;
    state.findings.push({
      type: 'question',
      title: lastQuestionResult.question,
      question: lastQuestionResult.question,
      answer: lastQuestionResult.answer,
      method: lastQuestionResult.method,
      code: lastQuestionResult.code,
    });
    renderFindings();
    const btn = document.getElementById('save-answer-btn');
    if (btn) { btn.textContent = 'Saved ✓'; btn.disabled = true; }
  });
}

function renderQaHistory() {
  const panel = document.getElementById('qa-history-panel');
  const host = document.getElementById('qa-history');
  panel.hidden = !state.qa.length;
  host.innerHTML = state.qa.map((item, index) => `
    <div class="qa-item">
      <button type="button" class="qa-q" data-qa-index="${index}">${escapeHtml(item.question)}</button>
      <div class="qa-a">${escapeHtml(item.answer)}</div>
    </div>`).join('');
  host.querySelectorAll('[data-qa-index]').forEach((btn) => {
    btn.addEventListener('click', () => runQuestion(state.qa[Number(btn.dataset.qaIndex)].question));
  });
}

function renderAskSuggestions() {
  const host = document.getElementById('ask-suggestions');
  const fromData = (state.analysis?.research_questions || []).slice(0, 3);
  const generic = ['How many rows and columns are there?', 'Are there duplicate rows?', 'What correlates most strongly?'];
  const all = [...fromData, ...generic];
  host.innerHTML = `<span class="hint">Try:</span> ` + all.map((q, i) => `<button type="button" class="chip suggestion" data-suggest="${i}">${escapeHtml(q)}</button>`).join('');
  host.querySelectorAll('[data-suggest]').forEach((btn) => btn.addEventListener('click', () => runQuestion(all[Number(btn.dataset.suggest)])));
}

document.getElementById('question-btn').addEventListener('click', () => runQuestion(questionInput.value));
questionInput.addEventListener('keydown', (event) => { if (event.key === 'Enter') runQuestion(questionInput.value); });

// ---- Client-side router: one page at a time, addressable by URL hash (#/visualization), so
// the browser back/forward buttons and bookmarks work. Pages that need a dataset show a
// "load a dataset first" gate instead of an empty, confusing screen.

const PAGE_ROUTES = ['datasets', 'overview', 'visualization', 'ask', 'export', 'changelog', 'explore', 'projects', 'analysis'];
const PAGE_HOOKS = {};   // page name -> function(param); social.js registers explore/projects/analysis and enriches datasets/overview
let routeParam = null;
const ROUTE_ALIASES = {workspace: 'visualization'}; // links from before the tab was renamed

function currentRoute() {
  const shared = window.location.hash.match(/^#\/a\/([\w-]+)/);
  routeParam = shared ? shared[1] : null;
  if (shared) return 'analysis';
  const match = window.location.hash.match(/^#\/(\w+)/);
  const route = match ? (ROUTE_ALIASES[match[1]] || match[1]) : 'datasets';
  return PAGE_ROUTES.includes(route) ? route : 'datasets';
}

function navigate(route) {
  if (window.location.hash === `#/${route}`) renderRoute();
  else window.location.hash = `#/${route}`;
}

function renderRoute() {
  const route = currentRoute();
  const target = document.querySelector(`.page[data-page="${route}"]`);
  const gated = target.dataset.requires === 'dataset' && !state.analysis;
  const shown = gated ? 'gate' : route;
  document.querySelectorAll('.page').forEach((el) => { el.hidden = el.dataset.page !== shown; });
  document.querySelectorAll('#site-nav a').forEach((link) => {
    const active = link.dataset.nav === (route === 'analysis' ? 'explore' : route);
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
  });
  document.title = `${target.dataset.title} · DataScopia`;
  window.scrollTo(0, 0);
  if (shown === 'datasets') renderSavedAnalysesPanel();
  if (shown === 'visualization') resizeAllCharts();
  if (shown === 'changelog') renderChangelog();
  if (shown === 'ask') renderAskSuggestions();
  if (shown === 'export') { renderExportForms(); renderExportSummary(); }
  if (PAGE_HOOKS[shown]) PAGE_HOOKS[shown](routeParam);
}
// ---- Changelog page: releases come from /api/changelog; the stats, category filters and search
// are all derived from that data, so adding a release to changelog.json is the only edit needed.

const changelog = {data: null, tag: 'All', query: '', loading: null};
const TAG_CLASS = {Design: 'tag-design', Platform: 'tag-platform', Analysis: 'tag-analysis', Foundation: 'tag-foundation', Data: 'tag-data', Export: 'tag-export', Origin: 'tag-origin'};

function loadChangelog() {
  if (changelog.data) return Promise.resolve(changelog.data);
  if (!changelog.loading) {
    changelog.loading = fetch(`${API_BASE}/api/changelog`)
      .then((response) => { if (!response.ok) throw new Error('unavailable'); return response.json(); })
      .then((data) => { changelog.data = data; return data; })
      .finally(() => { changelog.loading = null; });
  }
  return changelog.loading;
}

function formatReleaseDate(iso) {
  const date = new Date(`${iso}T00:00:00`);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleDateString(undefined, {month: 'short', day: 'numeric', year: 'numeric'});
}

async function renderChangelog() {
  const timeline = document.getElementById('cl-timeline');
  let data;
  try {
    data = await loadChangelog();
  } catch (error) {
    timeline.innerHTML = '<p class="hint">Could not load the release history. Is the API server running?</p>';
    return;
  }
  const stats = data.stats;
  document.getElementById('cl-stats').innerHTML = `
    <div class="cl-stat"><span>Releases</span><strong>${stats.releases}</strong><em>Documented milestones</em></div>
    <div class="cl-stat"><span>Latest version</span><strong>${escapeHtml(data.latest)}</strong><em>${escapeHtml(formatReleaseDate(stats.latest_date))}</em></div>
    <div class="cl-stat"><span>Days of building</span><strong>${stats.days}</strong><em>Since ${escapeHtml(formatReleaseDate(stats.first_date))}</em></div>
    <div class="cl-stat"><span>Changes shipped</span><strong>${stats.changes}</strong><em>Across all releases</em></div>`;

  const counts = {All: data.releases.length};
  data.releases.forEach((release) => { counts[release.tag] = (counts[release.tag] || 0) + 1; });
  const filters = document.getElementById('cl-filters');
  filters.innerHTML = ['All', ...data.tags].map((tag) => `
    <button type="button" class="cl-filter ${tag === changelog.tag ? 'active' : ''}" data-tag="${escapeHtml(tag)}">${escapeHtml(tag)} <span>${counts[tag] || 0}</span></button>`).join('');
  filters.querySelectorAll('[data-tag]').forEach((btn) => btn.addEventListener('click', () => {
    changelog.tag = btn.dataset.tag;
    renderChangelog();
  }));

  const query = changelog.query.trim().toLowerCase();
  const visible = data.releases.filter((release) => {
    if (changelog.tag !== 'All' && release.tag !== changelog.tag) return false;
    if (!query) return true;
    return [release.version, release.name, release.tag, release.meta, ...release.items].join(' ').toLowerCase().includes(query);
  });
  if (!visible.length) {
    timeline.innerHTML = '<div class="panel"><p class="hint">No releases match that filter.</p></div>';
    return;
  }
  timeline.innerHTML = visible.map((release, index) => {
    const isLatest = release.version === data.latest;
    return `
      <article class="cl-card ${isLatest ? 'current' : ''}" style="animation-delay: ${Math.min(index, 8) * 45}ms">
        <div class="cl-side">
          <div>
            <div class="cl-version">${escapeHtml(release.version)}</div>
            <div class="cl-date">${escapeHtml(formatReleaseDate(release.date))}</div>
          </div>
          <span class="cl-badge ${TAG_CLASS[release.tag] || ''}">${isLatest ? 'Latest' : escapeHtml(release.tag)}</span>
        </div>
        <div class="cl-body">
          <div class="cl-body-top">
            <h2>${escapeHtml(release.name)}</h2>
            <span class="cl-meta">${escapeHtml(release.meta || '')}</span>
          </div>
          <ul>${release.items.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>
        </div>
      </article>`;
  }).join('');
}

document.getElementById('cl-search').addEventListener('input', (event) => {
  changelog.query = event.target.value;
  renderChangelog();
});

// the header shows the current version, straight from the same data
loadChangelog().then((data) => {
  const badge = document.getElementById('nav-version');
  badge.textContent = data.latest;
  badge.hidden = false;
}).catch(() => { /* the badge is decoration; the app works without it */ });

// ---- First-run guidance: after a dataset is analyzed the user lands on the Overview, which can
// look like the whole app. A dismissible guide plus a short spotlight tour point out where the
// rest of the features live (Visualization, Ask, Export, saving). Shown automatically once per
// browser; the "Quick tour" button brings it back any time.

const ONBOARD_KEY = 'datascopia.onboarding.v1';
function onboardingSeen() {
  try { return localStorage.getItem(ONBOARD_KEY) === 'seen'; } catch (error) { return false; }
}
function markOnboardingSeen() {
  try { localStorage.setItem(ONBOARD_KEY, 'seen'); } catch (error) { /* private browsing */ }
}

const TOUR_STEPS = [
  {target: '[data-nav=overview]', title: 'Overview — you are here',
   body: 'Start here: how big the data is, a quality score with one-click fixes, and what every column means.'},
  {target: '[data-nav=visualization]', title: 'Visualization',
   body: 'Charts picked for your data (maps, time series, distributions), plus editors to filter and clean it. Every chart is drawn from all rows.'},
  {target: '[data-nav=ask]', title: 'Ask',
   body: 'Ask a question in plain English, like "which region has the highest income?". Answers are computed from your data, never guessed.'},
  {target: '[data-nav=export]', title: 'Export',
   body: 'Turn the session into a runnable Jupyter notebook (with your cleaning steps replayed), a PDF report, or the quality report.'},
  {target: '#save-analysis-btn', title: 'Save your work',
   body: 'Save this analysis in your browser and pick it up again later from the Datasets page.'},
];
const tour = {index: 0, active: false, els: null};

function showWelcomeCard(force = false) {
  const card = document.getElementById('welcome-card');
  card.hidden = false;
  if (force) card.scrollIntoView({behavior: 'smooth', block: 'start'});
}
function hideWelcomeCard() {
  document.getElementById('welcome-card').hidden = true;
  markOnboardingSeen();
}

function ensureTourElements() {
  if (tour.els) return tour.els;
  const backdrop = document.createElement('div');
  backdrop.className = 'tour-spot';
  const pop = document.createElement('div');
  pop.className = 'tour-pop';
  pop.setAttribute('role', 'dialog');
  pop.setAttribute('aria-live', 'polite');
  document.body.append(backdrop, pop);
  tour.els = {backdrop, pop};
  return tour.els;
}

function positionTour() {
  if (!tour.active) return;
  const step = TOUR_STEPS[tour.index];
  const target = document.querySelector(step.target);
  const {backdrop, pop} = tour.els;
  if (!target || target.offsetParent === null) {
    // target not on screen (e.g. hidden button): skip to the next step instead of pointing at nothing
    nextTourStep(1);
    return;
  }
  const rect = target.getBoundingClientRect();
  const pad = 6;
  Object.assign(backdrop.style, {
    left: `${rect.left - pad}px`, top: `${rect.top - pad}px`,
    width: `${rect.width + pad * 2}px`, height: `${rect.height + pad * 2}px`,
  });
  const width = Math.min(340, window.innerWidth - 24);
  pop.style.width = `${width}px`;
  const left = Math.max(12, Math.min(window.innerWidth - width - 12, rect.left + rect.width / 2 - width / 2));
  pop.style.left = `${left}px`;
  pop.style.top = `${rect.bottom + 18}px`;
  pop.style.setProperty('--arrow-x', `${Math.max(18, Math.min(width - 18, rect.left + rect.width / 2 - left))}px`);
}

function renderTourStep() {
  const step = TOUR_STEPS[tour.index];
  const {pop} = ensureTourElements();
  const last = tour.index === TOUR_STEPS.length - 1;
  pop.innerHTML = `
    <div class="tour-count">Step ${tour.index + 1} of ${TOUR_STEPS.length}</div>
    <h3>${escapeHtml(step.title)}</h3>
    <p>${escapeHtml(step.body)}</p>
    <div class="tour-dots">${TOUR_STEPS.map((_, i) => `<span class="${i === tour.index ? 'on' : ''}"></span>`).join('')}</div>
    <div class="tour-actions">
      <button type="button" class="tour-skip" data-tour="skip">Skip tour</button>
      <span>
        ${tour.index > 0 ? '<button type="button" class="btn secondary card-small-btn" data-tour="back">Back</button>' : ''}
        <button type="button" class="btn card-small-btn" data-tour="next">${last ? 'Finish' : 'Next'}</button>
      </span>
    </div>`;
  pop.querySelector('[data-tour=skip]').addEventListener('click', endTour);
  pop.querySelector('[data-tour=next]').addEventListener('click', () => (last ? endTour() : nextTourStep(1)));
  pop.querySelector('[data-tour=back]')?.addEventListener('click', () => nextTourStep(-1));
  positionTour();
}

function nextTourStep(delta) {
  const next = tour.index + delta;
  if (next < 0) return;
  if (next >= TOUR_STEPS.length) { endTour(); return; }
  tour.index = next;
  renderTourStep();
}

function startTour() {
  if (!state.analysis) return;
  navigate('overview');
  window.scrollTo(0, 0);
  tour.index = 0;
  tour.active = true;
  ensureTourElements();
  document.body.classList.add('touring');
  renderTourStep();
}

function endTour() {
  tour.active = false;
  document.body.classList.remove('touring');
  markOnboardingSeen();
}

function startOnboarding() {
  if (onboardingSeen()) return; // returning users get the guide only via the "Quick tour" button
  showWelcomeCard();
  setTimeout(startTour, 700); // let the Overview render before pointing at things
}

document.getElementById('welcome-dismiss').addEventListener('click', hideWelcomeCard);
document.getElementById('welcome-dismiss-2').addEventListener('click', hideWelcomeCard);
document.getElementById('welcome-tour-btn').addEventListener('click', startTour);
document.getElementById('tour-btn').addEventListener('click', () => { showWelcomeCard(true); startTour(); });
document.querySelectorAll('[data-step-scroll]').forEach((link) => link.addEventListener('click', (event) => {
  event.preventDefault();
  document.getElementById(link.dataset.stepScroll)?.scrollIntoView({behavior: 'smooth', block: 'start'});
}));
window.addEventListener('resize', positionTour);
document.addEventListener('keydown', (event) => {
  if (!tour.active) return;
  if (event.key === 'Escape') endTour();
  if (event.key === 'ArrowRight') nextTourStep(1);
  if (event.key === 'ArrowLeft') nextTourStep(-1);
});

window.addEventListener('hashchange', renderRoute);
renderSavedAnalysesPanel();
renderFindings();
renderRoute();
