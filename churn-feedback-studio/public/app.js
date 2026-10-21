'use strict';

const MAX_FILE_BYTES = 3 * 1024 * 1024;
const state = { file: null, preview: null, result: null, visibleRows: 100, filteredRows: [] };
const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}
function pct(value, digits = 1) { return value == null || !Number.isFinite(Number(value)) ? '—' : `${(Number(value) * 100).toFixed(digits)}%`; }
function number(value, digits = 2) { return value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(digits); }
function intFmt(value) { return new Intl.NumberFormat('en-US').format(Number(value || 0)); }
function display(value, fallback = '—') { return value == null || value === '' ? fallback : String(value); }
function notify(message) { const toast = $('toast'); toast.textContent = message; toast.classList.add('show'); window.clearTimeout(notify.timer); notify.timer = window.setTimeout(() => toast.classList.remove('show'), 3200); }
function setMessage(message, type = '') { const node = $('uploadMessage'); node.textContent = message || ''; node.className = `message ${type}`.trim(); }
function setLoading(visible, title = 'Analysing customer signals', sub = 'Training and evaluating the churn model…') {
  $('loadingOverlay').classList.toggle('hidden', !visible);
  $('loadingTitle').textContent = title;
  $('loadingText').textContent = sub;
  $('analyzeButton').disabled = visible || !state.file || !$('targetColumn').value;
  $('demoButton').disabled = visible;
  $('emptyDemoButton').disabled = visible;
}
async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { throw new Error(`The server returned an unreadable response (HTTP ${response.status}).`); }
  if (!response.ok) throw new Error(data.detail || data.message || `Request failed with HTTP ${response.status}.`);
  return data;
}
function optionsHtml(columns, blankLabel, blankValue = '') {
  return `<option value="">${escapeHtml(blankLabel)}</option>` + columns.map((col) => `<option value="${escapeHtml(col)}">${escapeHtml(col)}</option>`).join('');
}
function setSelectOptions(id, columns, blankLabel, selectedValue = '') {
  const select = $(id);
  select.innerHTML = optionsHtml(columns, blankLabel);
  select.disabled = false;
  if (selectedValue && columns.includes(selectedValue)) select.value = selectedValue;
  else select.value = '';
}
async function handleFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.csv')) { setMessage('Please choose a CSV file. Excel workbooks are not supported.', 'error'); return; }
  if (file.size > MAX_FILE_BYTES) { setMessage('This Vercel-friendly build accepts CSV files up to 3 MiB. Reduce the file before uploading.', 'error'); return; }
  state.file = file;
  state.preview = null;
  $('fileTitle').textContent = file.name;
  $('fileSubtitle').textContent = `${(file.size / 1024).toFixed(1)} KB · checking columns…`;
  $('analyzeButton').disabled = true;
  setMessage('Inspecting the CSV header and checking for churn, feedback, ID and date columns…');
  try {
    const form = new FormData(); form.append('file', file);
    const preview = await requestJson('/api/preview', { method: 'POST', body: form });
    state.preview = preview;
    setSelectOptions('targetColumn', preview.columns, 'Select churn target', preview.target_suggestion);
    setSelectOptions('textColumn', preview.columns, 'No feedback text', preview.text_suggestion);
    setSelectOptions('idColumn', preview.columns, 'Generate row IDs', preview.customer_id_suggestion);
    setSelectOptions('dateColumn', preview.columns, 'No cohort date', preview.date_suggestion);
    $('fileSubtitle').textContent = `${intFmt(preview.row_count_preview)}${preview.row_limit_exceeded ? '+' : ''} rows previewed · ${preview.columns.length} columns`;
    $('analyzeButton').disabled = !preview.target_suggestion;
    if (preview.row_limit_exceeded) {
      setMessage(`The file has at least ${intFmt(preview.max_rows + 1)} rows. Please reduce it to ${intFmt(preview.max_rows)} rows or fewer before analysis.`, 'error');
      $('analyzeButton').disabled = true;
    } else if (!preview.target_suggestion) {
      setMessage('No obvious churn target was detected. Select the correct target from the dropdown after reviewing your column names.', '');
      $('analyzeButton').disabled = false;
    } else {
      setMessage(`Ready to analyse ${file.name}. Verify the target and feedback columns before running the model.`, 'success');
    }
  } catch (error) {
    state.file = null;
    $('fileTitle').textContent = 'Drop your customer CSV here';
    $('fileSubtitle').textContent = 'or click to browse · CSV only';
    setMessage(error.message, 'error');
  }
}

$('csvFile').addEventListener('change', (event) => handleFile(event.target.files?.[0]));
const dropzone = $('dropzone');
dropzone.addEventListener('dragover', (event) => { event.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (event) => { event.preventDefault(); dropzone.classList.remove('dragover'); const file = event.dataTransfer?.files?.[0]; if (file) handleFile(file); });
$('targetColumn').addEventListener('change', () => { $('analyzeButton').disabled = !state.file || !$('targetColumn').value; });
$('analyzeButton').addEventListener('click', runUploadedAnalysis);
$('demoButton').addEventListener('click', runDemo);
$('emptyDemoButton').addEventListener('click', runDemo);
$('customerSearch').addEventListener('input', () => { state.visibleRows = 100; renderCustomerRows(); });
$('loadMoreButton').addEventListener('click', () => { state.visibleRows += 100; renderCustomerRows(); });
$('downloadButton').addEventListener('click', downloadPredictions);

async function runUploadedAnalysis() {
  if (!state.file) { setMessage('Choose a CSV file first.', 'error'); return; }
  const target = $('targetColumn').value;
  if (!target) { setMessage('Select the binary churn target column.', 'error'); return; }
  const form = new FormData();
  form.append('file', state.file);
  form.append('target_column', target);
  form.append('text_column', $('textColumn').value);
  form.append('customer_id_column', $('idColumn').value);
  form.append('date_column', $('dateColumn').value);
  setLoading(true, 'Training and evaluating the model', 'Calibrating probabilities, mining feedback and ranking retention actions…');
  setMessage('Analysis is running. The model uses only labelled rows for training and the holdout evaluation.');
  try {
    const result = await requestJson('/api/analyze', { method: 'POST', body: form });
    state.result = result; state.visibleRows = 100;
    renderResults(result);
    setMessage(`Analysis complete: ${intFmt(result.summary.total_customers)} customers scored.`, 'success');
    notify('Analysis completed successfully.');
  } catch (error) {
    setMessage(error.message, 'error');
    notify('Analysis could not be completed. Review the message beside the upload form.');
  } finally { setLoading(false); }
}

async function runDemo() {
  setLoading(true, 'Preparing the sample workspace', 'Generating synthetic customers and running the full analysis pipeline…');
  setMessage('Running the synthetic demo. No uploaded customer data is used.');
  try {
    const result = await requestJson('/api/demo', { method: 'POST' });
    state.result = result; state.visibleRows = 100; state.file = null;
    renderResults(result);
    setMessage('Synthetic demo loaded. Use a CSV to analyse your own historical customers.', 'success');
    notify('Sample analysis is ready.');
  } catch (error) {
    setMessage(error.message, 'error');
    notify('The demo failed. See the upload panel for details.');
  } finally { setLoading(false); }
}

function kpiCard(label, value, footer, symbol, tone = '') {
  return `<article class="kpi-card"><div class="kpi-label"><span>${escapeHtml(label)}</span><span class="kpi-icon">${escapeHtml(symbol)}</span></div><div class="kpi-value ${tone}">${value}</div><div class="kpi-foot">${footer}</div></article>`;
}
function modelMetric(label, value, foot) {
  return `<div class="model-metric"><span>${escapeHtml(label)}</span><b>${value}</b><small>${escapeHtml(foot)}</small></div>`;
}
function renderResults(data) {
  $('emptyState').classList.add('hidden');
  $('resultsArea').classList.remove('hidden');
  $('sourceBadge').textContent = data.demo ? 'SYNTHETIC DEMO' : 'CUSTOM DATASET';
  const s = data.summary || {};
  const metrics = data.metrics || {};
  const feedback = data.feedback_summary || {};
  $('kpiGrid').innerHTML = [
    kpiCard('Observed churn rate', pct(s.observed_churn_rate), `${intFmt(s.labelled_customers)} labelled · ${intFmt(s.unlabelled_customers)} to score`, '↗', 'risk-value'),
    kpiCard('High-risk customers', intFmt(s.high_risk_customers), `${intFmt(s.critical_risk_customers)} in the critical risk tier`, '!', 'risk-value'),
    kpiCard('Holdout ROC AUC', number(metrics.roc_auc, 3), `${intFmt(s.test_customers)} unseen labelled customers`, '◎'),
    kpiCard('Negative feedback', feedback.negative_rate == null ? '—' : pct(feedback.negative_rate), feedback.available ? `${intFmt(feedback.rows_with_feedback)} rows with feedback` : 'No text column selected', '✳')
  ].join('');
  $('modelMetricGrid').innerHTML = [
    modelMetric('ROC AUC', number(metrics.roc_auc, 3), 'Ranking quality · higher better'),
    modelMetric('Brier score', number(metrics.brier_score, 3), 'Probability error · lower better'),
    modelMetric('Expected calibration error', number(metrics.expected_calibration_error, 3), 'Bin error · lower better'),
    modelMetric('Recall @ 0.50', pct(metrics.recall), 'Churners found at 50% threshold')
  ].join('');
  $('modelComparisonBody').innerHTML = (data.models || []).map((m) => `<tr><td><span class="model-name"><span class="model-dot ${m.model.includes('Forest') ? 'rf' : ''}"></span>${escapeHtml(m.model)}</span></td><td>${number(m.roc_auc, 3)}</td><td>${number(m.average_precision, 3)}</td><td>${pct(m.recall)}</td><td>${number(m.f1, 3)}</td><td>${number(m.brier_score, 3)}</td><td>${escapeHtml(m.calibration)}</td></tr>`).join('');
  renderSentiment(feedback);
  renderTopics(data.topics || [], feedback);
  renderCalibration(data.calibration_bins || []);
  renderActions(data.actions || []);
  renderImportance(data.feature_importance || []);
  renderCohorts(data.cohorts || {});
  renderSegments(data.segments || []);
  state.filteredRows = Array.isArray(data.predictions) ? [...data.predictions].sort((a, b) => Number(b.churn_probability) - Number(a.churn_probability)) : [];
  $('customerSearch').value = '';
  renderCustomerRows();
  $('downloadButton').disabled = !state.filteredRows.length;
  $('resultFootnote').innerHTML = (data.notes || []).map((n) => `<div>• ${escapeHtml(n)}</div>`).join('');
  if (data.demo && data.source_note) $('resultFootnote').innerHTML = `<div class="demo-disclaimer"><b>Sample dataset:</b> ${escapeHtml(data.source_note)}</div>` + $('resultFootnote').innerHTML;
  // Smoothly bring the new results into view after a full analysis.
  $('snapshotHeading').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderSentiment(feedback) {
  const counts = feedback.counts || {};
  const order = ['Negative', 'Neutral', 'Positive', 'No feedback'];
  const total = Object.values(counts).reduce((sum, value) => sum + Number(value || 0), 0) || 1;
  const classMap = { Negative: 'negative', Neutral: 'neutral', Positive: 'positive', 'No feedback': 'no-feedback' };
  if (!feedback.available) {
    $('feedbackSubtitle').textContent = 'No feedback text column was selected; churn modelling still ran on structured features.';
    $('sentimentChart').innerHTML = '<div class="inline-empty">Select a complaint or feedback text column to enable sentiment analysis.</div>';
    $('sentimentSummary').innerHTML = '';
    return;
  }
  $('feedbackSubtitle').textContent = `Sentiment estimates from “${feedback.column}” using a lexicon-based scorer.`;
  $('sentimentChart').innerHTML = order.filter((label) => counts[label] != null).map((label) => {
    const count = Number(counts[label] || 0), ratio = count / total * 100;
    return `<div class="sentiment-row"><label>${escapeHtml(label)}</label><div class="bar-track"><div class="bar-fill ${classMap[label]}" style="width:${ratio.toFixed(2)}%"></div></div><strong>${intFmt(count)}</strong></div>`;
  }).join('');
  $('sentimentSummary').innerHTML = `<span class="summary-chip">Average tone <b>${number(feedback.average_compound, 2)}</b></span><span class="summary-chip">Feedback coverage <b>${pct(feedback.rows_with_feedback / Math.max(1, total))}</b></span>`;
}
function renderTopics(topics, feedback) {
  if (!feedback.available) { $('topicList').innerHTML = '<div class="inline-empty">Choose a feedback text column to extract recurring topics.</div>'; return; }
  if (!topics.length) { $('topicList').innerHTML = '<div class="inline-empty">There was not enough varied feedback text for stable topic extraction. Try a larger sample with more unique complaints.</div>'; return; }
  $('topicList').innerHTML = topics.map((topic, i) => `<div class="topic-row"><span class="topic-index">T${i + 1}</span><div><strong>${escapeHtml(topic.label)}</strong><div class="topic-tags">${(topic.keywords || []).slice(0, 5).map((term) => `<span class="topic-tag">${escapeHtml(term)}</span>`).join('')}</div></div></div>`).join('');
}
function renderCalibration(bins) {
  const width = 440, height = 270, left = 42, right = 18, top = 16, bottom = 39;
  const plotW = width - left - right, plotH = height - top - bottom;
  const x = (v) => left + Number(v) * plotW;
  const y = (v) => top + (1 - Number(v)) * plotH;
  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Calibration plot comparing predicted probability with observed churn rate"><title>Calibration reliability plot</title>`;
  [0, .25, .5, .75, 1].forEach((tick) => {
    svg += `<line class="calib-grid" x1="${left}" y1="${y(tick)}" x2="${width-right}" y2="${y(tick)}"/><line class="calib-grid" x1="${x(tick)}" y1="${top}" x2="${x(tick)}" y2="${height-bottom}"/><text class="calib-axis" x="${x(tick)}" y="${height-18}" text-anchor="middle">${tick.toFixed(2)}</text><text class="calib-axis" x="${left-8}" y="${y(tick)+3}" text-anchor="end">${tick.toFixed(2)}</text>`;
  });
  svg += `<line class="calib-diagonal" x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}"/>`;
  bins.forEach((point) => { svg += `<circle class="calib-point" cx="${x(point.mean_predicted)}" cy="${y(point.observed_churn)}" r="${Math.max(4, Math.min(8, 3 + Math.sqrt(point.count)))}"><title>${escapeHtml(point.bucket)} · predicted ${pct(point.mean_predicted)} · observed ${pct(point.observed_churn)} · n=${point.count}</title></circle>`; });
  svg += `<text class="calib-axis" x="${left + plotW/2}" y="${height-2}" text-anchor="middle">Mean predicted churn probability</text><text class="calib-axis" transform="translate(12 ${top+plotH/2}) rotate(-90)" text-anchor="middle">Observed churn rate</text></svg>`;
  if (!bins.length) svg = '<div class="inline-empty">Calibration data is not available for this sample.</div>';
  $('calibrationChart').innerHTML = svg;
}
function renderActions(actions) {
  if (!actions.length) { $('actionList').innerHTML = '<div class="inline-empty">No retention actions could be ranked for this dataset.</div>'; return; }
  const max = Math.max(...actions.map((a) => a.aggregate_priority || a.priority_score || 1), 1);
  $('actionList').innerHTML = actions.map((action, i) => {
    const width = 100 * (action.aggregate_priority || action.priority_score || 0) / max;
    return `<div class="action-row"><span class="action-rank">${String(i + 1).padStart(2, '0')}</span><div class="action-main"><b>${escapeHtml(action.action)}</b><p>${escapeHtml(action.reason)} · ${intFmt(action.customers_flagged)} customers flagged</p><div class="importance-track" style="margin-top:8px"><div class="importance-fill" style="width:${Math.max(3, width)}%"></div></div></div><div class="action-meta"><strong>${number(action.priority_score, 0)}</strong><small>mean priority /100</small></div></div>`;
  }).join('');
}
function renderImportance(features) {
  if (!features.length) { $('featureImportance').innerHTML = '<div class="inline-empty">Feature coefficients are not available for this fitted model.</div>'; return; }
  $('featureImportance').innerHTML = features.map((feature) => `<div class="importance-row"><span class="importance-name" title="${escapeHtml(feature.feature)}">${escapeHtml(feature.feature)}</span><div class="importance-track"><div class="importance-fill" style="width:${Math.max(1, Number(feature.relative_percent || 0))}%"></div></div><strong>${number(feature.relative_percent, 0)}</strong></div>`).join('');
}
function renderCohorts(cohorts) {
  $('cohortSource').textContent = cohorts.source || 'Cohort analysis needs a date or tenure column.';
  const rows = cohorts.rows || [];
  $('cohortBody').innerHTML = rows.map((row) => `<tr><td>${escapeHtml(row.cohort)}</td><td>${intFmt(row.customers)}</td><td class="rate-cell ${row.churn_rate >= .35 ? 'rate-high' : 'rate-low'}">${pct(row.churn_rate)}</td><td>${pct(row.retention_rate)}</td></tr>`).join('');
  $('cohortEmpty').classList.toggle('hidden', rows.length > 0);
  if (!rows.length) $('cohortEmpty').textContent = cohorts.note || 'Choose a signup date or tenure column to unlock cohort analysis.';
}
function renderSegments(segments) {
  if (!segments.length) { $('segmentList').innerHTML = '<div class="inline-empty">At least two useful numeric features and 18 rows are needed for KMeans segments.</div>'; return; }
  const maxCount = Math.max(...segments.map((s) => s.customers), 1);
  $('segmentList').innerHTML = segments.map((segment) => `<div class="segment-row"><div class="segment-head"><strong>${escapeHtml(segment.segment)}</strong><span>${intFmt(segment.customers)} customers</span></div><div class="segment-stats"><div class="segment-stat"><label>Average model risk</label><b>${pct(segment.average_risk)}</b></div><div class="segment-stat"><label>Observed churn</label><b>${pct(segment.observed_churn_rate)}</b></div><div class="segment-stat"><label>Share of file</label><b>${pct(segment.customers / Math.max(1, state.result?.summary?.total_customers || 1))}</b></div></div><div class="segment-bar"><span style="width:${100 * segment.customers / maxCount}%"></span></div></div>`).join('');
}
function renderCustomerRows() {
  if (!state.result) return;
  const search = $('customerSearch').value.trim().toLowerCase();
  const allRows = state.filteredRows || [];
  const rows = allRows.filter((row) => [row.customer_id, row.risk_tier, row.sentiment, row.topic, row.recommended_action, row.actual_status].some((value) => String(value ?? '').toLowerCase().includes(search)));
  $('customerCount').textContent = `${intFmt(rows.length)} customer${rows.length === 1 ? '' : 's'}`;
  const pageRows = rows.slice(0, state.visibleRows);
  $('customerTableBody').innerHTML = pageRows.map((row) => {
    const prob = Number(row.churn_probability || 0);
    const tierClass = String(row.risk_tier || 'low').toLowerCase();
    const sentiment = String(row.sentiment || 'No feedback');
    return `<tr><td class="customer-name">${escapeHtml(row.customer_id)}</td><td>${row.actual_status ? `<span class="actual-pill">${escapeHtml(row.actual_status)}</span>` : '<span class="actual-pill">Unlabelled</span>'}</td><td class="risk-cell"><div class="risk-line"><strong>${pct(prob)}</strong><span class="risk-bar"><span style="width:${Math.round(prob*100)}%"></span></span></div><span class="risk-pill ${escapeHtml(tierClass)}">${escapeHtml(row.risk_tier)}</span></td><td><span class="sentiment-label ${escapeHtml(sentiment.toLowerCase())}">${escapeHtml(sentiment)}</span></td><td class="topic-cell" title="${escapeHtml(row.topic || '')}">${escapeHtml(row.topic || '—')}</td><td class="action-cell" title="${escapeHtml(row.action_reason || '')}">${escapeHtml(row.recommended_action || '—')}</td></tr>`;
  }).join('');
  if (!pageRows.length) $('customerTableBody').innerHTML = '<tr><td colspan="6" style="text-align:center;padding:28px;color:#909990">No customers match this search.</td></tr>';
  $('tableFooterText').textContent = pageRows.length ? `Showing ${intFmt(pageRows.length)} of ${intFmt(rows.length)} matching customers, sorted by churn risk.` : 'No rows to show.';
  $('loadMoreButton').classList.toggle('hidden', rows.length <= pageRows.length);
  state.visibleRows = Math.min(state.visibleRows, Math.max(100, rows.length));
}
function downloadPredictions() {
  if (!state.result?.predictions?.length) return;
  const fields = ['customer_id','record_type','actual_status','churn_probability','risk_tier','predicted_churn_at_50pct','sentiment','sentiment_score','topic','segment','recommended_action','action_reason','next_best_action'];
  const quote = (value) => { const text = value == null ? '' : String(value); return /[\r\n",]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text; };
  const csv = [fields.join(','), ...state.result.predictions.map((row) => fields.map((field) => quote(row[field])).join(','))].join('\r\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = 'churn_risk_and_retention_actions.csv'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
  notify('Prediction CSV downloaded.');
}

// Make the sticky navigation state reflect the current section, without mutating history.
const navLinks = Array.from(document.querySelectorAll('.nav-link'));
const watchedSections = ['overview', 'feedback', 'customers', 'retention'].map((id) => $(id)).filter(Boolean);
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    navLinks.forEach((link) => link.classList.toggle('active', link.getAttribute('href') === `#${visible.target.id}`));
  }, { rootMargin: '-18% 0px -66% 0px', threshold: [0, .1, .3] });
  watchedSections.forEach((section) => observer.observe(section));
}

// Lightweight environment check; failure is not fatal in local preview.
requestJson('/api/health').then((data) => { if (data.status !== 'ok') throw new Error('Unexpected API status'); }).catch(() => { setMessage('API is not reachable yet. Deploy the project to Vercel or run `uvicorn app:app --reload` from the project directory.', ''); });
