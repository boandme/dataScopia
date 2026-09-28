// Accounts, cloud projects, verified/shareable analyses, and the dataset graph.
// Loaded after app.js and built on its globals (state, navigate, renderAnalysis, chartSlots, ...).

const $ = (id) => document.getElementById(id);
state.user = null;
state.projectId = null;

// ---------------------------------------------------------------- small helpers

async function api(path, {method = 'GET', body} = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: body ? {'Content-Type': 'application/json'} : undefined,
    body: body ? JSON.stringify(body) : undefined,
    credentials: 'same-origin',
  });
  let data = null;
  try { data = await response.json(); } catch (error) { data = null; }
  if (!response.ok) {
    const err = new Error((data && data.detail) || `Request failed (${response.status}).`);
    err.status = response.status;
    throw err;
  }
  return data;
}

function toast(message, kind = 'ok') {
  const el = document.createElement('div');
  el.className = `toast toast-${kind}`;
  el.textContent = message;
  $('toast-host').appendChild(el);
  setTimeout(() => el.classList.add('out'), 3800);
  setTimeout(() => el.remove(), 4300);
}

function timeAgo(seconds) {
  const diff = Math.max(0, Date.now() / 1000 - seconds);
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`;
  if (diff < 86400 * 30) return `${Math.floor(diff / 86400)} d ago`;
  return new Date(seconds * 1000).toLocaleDateString(undefined, {month: 'short', day: 'numeric', year: 'numeric'});
}

function showError(el, message) {
  el.textContent = message;
  el.hidden = !message;
}

function openOverlay(id) { $(id).hidden = false; }
function closeOverlay(id) { $(id).hidden = true; }
function bindOverlayClose(id, closeIds) {
  closeIds.forEach((btn) => $(btn)?.addEventListener('click', () => closeOverlay(id)));
  $(id).addEventListener('click', (event) => { if (event.target === $(id)) closeOverlay(id); });
}
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') ['auth-overlay', 'settings-overlay', 'publish-overlay', 'join-overlay'].forEach((id) => { if (!$(id).hidden) closeOverlay(id); });
});
function bindSeg(container, onChange) {
  container.addEventListener('click', (event) => {
    const btn = event.target.closest('button[data-value]');
    if (!btn) return;
    container.querySelectorAll('button').forEach((b) => b.classList.toggle('on', b === btn));
    if (onChange) onChange(btn.dataset.value);
  });
}
const segValue = (container) => container.querySelector('button.on')?.dataset.value;

// ================================================================= accounts

const auth = {mode: 'login', onDone: null, onLocal: null};
// Firebase Authentication (loaded by firebase-client.js) resolves this with its API, or null in built-in-accounts mode.
const firebaseReady = new Promise((resolve) => { window.__resolveFirebase = resolve; setTimeout(() => resolve(null), 12000); });
let authConfig = {mode: 'local'};
const usingFirebase = () => authConfig.mode === 'firebase';

async function refreshUser() {
  try { authConfig = await api('/api/auth/config'); } catch (error) { authConfig = {mode: 'local'}; }
  try {
    state.user = (await api('/api/auth/me')).user;
  } catch (error) {
    state.user = null;
  }
  if (!state.user && usingFirebase()) {
    // Firebase may still remember the user even though our session cookie is gone: sign in silently.
    const fb = await firebaseReady;
    if (fb) { try { state.user = await fb.restoreSession(); } catch (error) { state.user = null; } }
  }
  renderAuthUI();
}

function renderAuthUI() {
  const user = state.user;
  $('auth-btn').hidden = Boolean(user);
  $('account-menu').hidden = !user;
  if (user) {
    const initial = (user.display_name || '?').trim().charAt(0).toUpperCase();
    $('account-name').textContent = user.display_name;
    $('account-avatar').textContent = initial;
    $('account-dd-avatar').textContent = initial;
    $('account-dd-name').textContent = user.display_name;
    $('account-dd-email').textContent = user.email;
  }
  updatePublishAvailability();
}

function setAuthMode(mode) {
  auth.mode = mode;
  document.querySelectorAll('[data-auth-tab]').forEach((b) => b.classList.toggle('on', b.dataset.authTab === mode));
  const register = mode === 'register';
  $('auth-name-row').hidden = !register;
  $('auth-hint').hidden = !register;
  $('auth-title').textContent = register ? 'Create your account' : 'Sign in to DataScopia';
  $('auth-submit').textContent = register ? 'Create account' : 'Sign in';
  $('auth-password').autocomplete = register ? 'new-password' : 'current-password';
  $('auth-google-block').hidden = !usingFirebase();
  $('auth-forgot').hidden = !usingFirebase() || register;
  showError($('auth-error'), '');
}

function openAuth({reason, mode = 'login', onDone = null, onLocal = null} = {}) {
  auth.onDone = onDone;
  auth.onLocal = onLocal;
  $('auth-reason').textContent = reason || 'Save projects, publish verified analyses and pick up where you left off on any device.';
  $('auth-local').hidden = !onLocal;
  setAuthMode(mode);
  openOverlay('auth-overlay');
  setTimeout(() => $('auth-email').focus(), 30);
}
window.openAuth = openAuth;

function requireUser(reason, action) {
  if (state.user) { action(); return true; }
  openAuth({reason, onDone: action});
  return false;
}

function finishSignIn(created) {
  $('auth-password').value = '';
  closeOverlay('auth-overlay');
  renderAuthUI();
  toast(created ? `Welcome, ${state.user.display_name}!` : `Signed in as ${state.user.display_name}`);
  const done = auth.onDone;
  auth.onDone = null;
  if (done) done();
  renderRoute();
}

$('auth-google').addEventListener('click', async () => {
  showError($('auth-error'), '');
  try {
    const fb = await firebaseReady;
    if (!fb) throw new Error('Google sign-in could not load (are you offline?).');
    state.user = await fb.signInWithGoogle();
    finishSignIn(false);
  } catch (error) { showError($('auth-error'), error.message); }
});
$('auth-forgot').addEventListener('click', async () => {
  const email = $('auth-email').value.trim();
  if (!email) { showError($('auth-error'), 'Enter your email above first, then click "Forgot your password?".'); return; }
  try {
    const fb = await firebaseReady;
    await fb.resetPassword(email);
    showError($('auth-error'), '');
    toast(`If ${email} has an account, a reset link is on its way.`);
  } catch (error) { showError($('auth-error'), error.message); }
});

document.querySelectorAll('[data-auth-tab]').forEach((b) => b.addEventListener('click', () => setAuthMode(b.dataset.authTab)));
$('auth-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  showError($('auth-error'), '');
  const submit = $('auth-submit');
  submit.disabled = true;
  submit.classList.add('loading');
  try {
    const email = $('auth-email').value.trim();
    const password = $('auth-password').value;
    const name = $('auth-name').value.trim();
    if (usingFirebase()) {
      const fb = await firebaseReady;
      if (!fb) throw new Error('Sign-in could not load (are you offline?). Check your connection and reload.');
      if (auth.mode === 'register' && password.length < 8) throw new Error('Choose a password of at least 8 characters.');
      state.user = auth.mode === 'register' ? await fb.signUp(email, password, name) : await fb.signIn(email, password);
    } else {
      const body = {email, password};
      if (auth.mode === 'register') body.display_name = name;
      state.user = (await api(`/api/auth/${auth.mode}`, {method: 'POST', body})).user;
    }
    finishSignIn(auth.mode === 'register');
  } catch (error) {
    showError($('auth-error'), error.message);
  } finally {
    submit.disabled = false;
    submit.classList.remove('loading');
  }
});
$('auth-local').addEventListener('click', () => { closeOverlay('auth-overlay'); const cb = auth.onLocal; auth.onLocal = null; if (cb) cb(); });
bindOverlayClose('auth-overlay', ['auth-close']);
$('auth-btn').addEventListener('click', () => openAuth());
$('auth-eye').addEventListener('click', () => {
  const input = $('auth-password');
  const show = input.type === 'password';
  input.type = show ? 'text' : 'password';
  $('auth-eye').classList.toggle('is-visible', show);
  $('auth-eye').setAttribute('aria-label', show ? 'Hide password' : 'Show password');
});

// account dropdown
$('account-chip').addEventListener('click', (event) => {
  event.stopPropagation();
  const dd = $('account-dropdown');
  dd.hidden = !dd.hidden;
  $('account-chip').setAttribute('aria-expanded', String(!dd.hidden));
});
document.addEventListener('click', (event) => { if (!event.target.closest('#account-menu')) $('account-dropdown').hidden = true; });
$('account-dropdown').addEventListener('click', (event) => { if (event.target.closest('a')) $('account-dropdown').hidden = true; });
$('signout-btn').addEventListener('click', async () => {
  $('account-dropdown').hidden = true;
  await api('/api/auth/logout', {method: 'POST', body: {}});
  if (usingFirebase()) { const fb = await firebaseReady; if (fb) await fb.signOut(); }
  state.user = null;
  state.projectId = null;
  renderAuthUI();
  $('save-status').hidden = true;
  toast('Signed out');
  renderRoute();
});

// settings
$('account-settings-btn').addEventListener('click', () => {
  $('account-dropdown').hidden = true;
  $('settings-name').value = state.user?.display_name || '';
  ['settings-current', 'settings-new', 'settings-delete-password'].forEach((id) => { $(id).value = ''; });
  showError($('settings-message'), '');
  firebaseReady.then((fb) => {
    const managed = usingFirebase() && fb;
    const google = managed && !fb.hasPassword();
    $('settings-password-block').hidden = google;                       // Google accounts have no password here
    $('settings-delete-password').closest('label').hidden = google;    // ...and confirm deletion in a Google pop-up instead
  });
  openOverlay('settings-overlay');
});
bindOverlayClose('settings-overlay', ['settings-close']);
function settingsMessage(text, ok = false) {
  const el = $('settings-message');
  el.textContent = text;
  el.hidden = !text;
  el.style.color = ok ? 'var(--accent-2)' : 'var(--danger)';
}
$('settings-save-name').addEventListener('click', async () => {
  try {
    const name = $('settings-name').value;
    const fb = usingFirebase() ? await firebaseReady : null;
    state.user = fb ? await fb.updateName(name) : (await api('/api/auth/profile', {method: 'PUT', body: {display_name: name}})).user;
    renderAuthUI();
    settingsMessage('Name updated.', true);
  } catch (error) { settingsMessage(error.message); }
});
$('settings-change-password').addEventListener('click', async () => {
  try {
    const fb = usingFirebase() ? await firebaseReady : null;
    if (fb) {
      if ($('settings-new').value.length < 8) throw new Error('Choose a password of at least 8 characters.');
      await fb.changePassword($('settings-current').value, $('settings-new').value);
    } else {
      await api('/api/auth/password', {method: 'POST', body: {current_password: $('settings-current').value, new_password: $('settings-new').value}});
    }
    ['settings-current', 'settings-new'].forEach((id) => { $(id).value = ''; });
    settingsMessage(fb ? 'Password changed.' : 'Password changed. Other devices were signed out.', true);
  } catch (error) { settingsMessage(error.message); }
});
$('settings-delete').addEventListener('click', async () => {
  if (!confirm('Delete your account, projects and published analyses permanently?')) return;
  try {
    const fb = usingFirebase() ? await firebaseReady : null;
    if (fb) await fb.deleteAccount($('settings-delete-password').value);
    else await api('/api/auth/delete', {method: 'POST', body: {password: $('settings-delete-password').value}});
    state.user = null;
    state.projectId = null;
    closeOverlay('settings-overlay');
    renderAuthUI();
    toast('Your account was deleted');
    navigate('datasets');
  } catch (error) { settingsMessage(error.message); }
});

// ================================================================= projects (cloud saves)

let saveMode = 'local';
let saveTimer = null;

function projectPayload() {
  return {
    dataset_id: state.dataset?.dataset_id,
    dataset_name: state.analysis?.dataset_name,
    dataset: state.dataset,
    findings: state.findings,
    transform_log: state.transformLog.filter((entry) => entry && entry.type),
    chart_configs: chartSlots.map((slot) => ({label: slot.label, ...chartConfigOf(slot)})),
  };
}

function setSaveStatus(text, kind = 'ok') {
  const el = $('save-status');
  el.hidden = !text;
  el.textContent = text;
  el.className = `save-status ${kind}`;
}

function openSaveNameModal() {
  $('save-name-input').value = state.analysis?.dataset_name ? `${state.analysis.dataset_name} analysis` : '';
  $('save-name-overlay').hidden = false;
  $('save-name-input').focus();
}

// called by app.js when "Save analysis" is clicked; returning true means "handled here"
window.handleSaveClick = function handleSaveClick() {
  if (state.user) {
    saveMode = 'cloud';
    if (state.projectId) { autosave(true); return true; }  // already a project: just save now
    return false;                                           // new project: let the naming dialog open
  }
  openAuth({
    reason: 'Sign in to save this analysis to your account and open it from any device.',
    onDone: () => { saveMode = 'cloud'; openSaveNameModal(); },
    onLocal: () => { saveMode = 'local'; openSaveNameModal(); },
  });
  return true;
};

window.cloudSaveConfirm = function cloudSaveConfirm(name) {
  if (saveMode !== 'cloud' || !state.user) { saveMode = 'local'; return false; }
  saveMode = 'local';
  createProject(name);
  return true;
};

async function createProject(name) {
  setSaveStatus('Saving...', 'busy');
  try {
    const {project} = await api('/api/projects', {method: 'POST', body: {...projectPayload(), name: (name || '').trim()}});
    state.projectId = project.id;
    state.projectName = project.name;
    setSaveStatus(`Saved ✓ ${new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})}`);
    toast(`Saved “${project.name}” to your account`);
  } catch (error) {
    setSaveStatus('Not saved', 'bad');
    toast(error.message, 'bad');
  }
}

async function autosave(announce = false) {
  if (!state.user || !state.projectId) return;
  setSaveStatus('Saving...', 'busy');
  try {
    const p = projectPayload();
    await api(`/api/projects/${state.projectId}`, {method: 'PUT', body: {findings: p.findings, transform_log: p.transform_log, chart_configs: p.chart_configs}});
    setSaveStatus(`Saved ✓ ${new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'})}`);
    if (announce) toast('Project saved');
  } catch (error) {
    setSaveStatus('Not saved', 'bad');
    if (announce) toast(error.message, 'bad');
  }
}

window.markProjectDirty = function markProjectDirty() {
  if (!state.user || !state.projectId || Date.now() < (state.suppressDirtyUntil || 0)) return;
  clearTimeout(saveTimer);
  setSaveStatus('Unsaved changes', 'busy');
  saveTimer = setTimeout(() => autosave(false), 2000);
};

// Put a stored/derived dataset into the workspace exactly as if it had just been analysed.
function applyWorkspacePayload(payload, {findings = [], chartConfigs = null, projectId = null, projectName = null} = {}) {
  clearTimeout(saveTimer);
  state.suppressDirtyUntil = Date.now() + 3000;
  state.dataset = payload.dataset;
  state.analysis = payload.analysis;
  state.slug = payload.dataset.slug || null;
  state.transformLog = payload.transform_log || [];
  state.findings = findings;
  state.projectId = projectId;
  state.projectName = projectName;
  state.qa = [];
  renderQaHistory();
  downloadFullBtn.hidden = !state.slug;
  fullAnalysisBtn.hidden = true;
  resetTransformBtn.hidden = state.transformLog.length === 0;
  const fields = Object.keys((state.analysis.preview || [])[0] || {});
  setVizSectionVisible(fields.length > 0);
  if (fields.length) {
    rebuildChartSlots(fields, state.analysis.recommendations || [], chartConfigs && chartConfigs.length ? chartConfigs : undefined);
    resetDatasetEditorSlots();
  }
  renderAnalysis({rebuildCharts: false});
  renderTransformLog();
  renderFindings();
  graph.scannedFor = null;
}

async function openProject(id) {
  try {
    const payload = await api(`/api/projects/${id}/open`, {method: 'POST', body: {}});
    applyWorkspacePayload(payload, {findings: payload.state.findings, chartConfigs: payload.state.chart_configs, projectId: id, projectName: payload.project.name});
    setSaveStatus(`Saved ✓ ${timeAgo(payload.project.updated_at)}`);
    (payload.warnings || []).forEach((w) => toast(w, 'bad'));
    toast(`Opened “${payload.project.name}”`);
    navigate('overview');
    return payload;
  } catch (error) {
    toast(error.message, 'bad');
    return null;
  }
}

function signInPrompt(el, text) {
  el.innerHTML = `<div class="panel gate-panel"><h2>Sign in required</h2><p class="hint">${escapeHtml(text)}</p><button class="btn" type="button" data-signin>Sign in or create an account</button></div>`;
  el.querySelector('[data-signin]').addEventListener('click', () => openAuth({reason: text}));
}

PAGE_HOOKS.projects = async function renderProjectsPage() {
  const body = $('projects-body');
  if (!state.user) { signInPrompt(body, 'Sign in to see the analyses saved to your account.'); return; }
  body.innerHTML = '<p class="hint">Loading your projects...</p>';
  try {
    const [{projects, limit}, mine] = await Promise.all([api('/api/projects'), api('/api/shares/mine')]);
    const cards = projects.map((p) => `
      <article class="project-card" data-project="${p.id}">
        <div class="project-top">
          <h3>${escapeHtml(p.name)}</h3>
          <span class="chip">${escapeHtml(p.source)}</span>
        </div>
        <p class="hint">${escapeHtml(p.dataset_name)} · ${p.rows.toLocaleString()} rows × ${p.cols} columns</p>
        <div class="project-stats"><span>${p.steps} cleaning step${p.steps === 1 ? '' : 's'}</span><span>${p.findings} finding${p.findings === 1 ? '' : 's'}</span><span>${p.charts} chart${p.charts === 1 ? '' : 's'}</span></div>
        <div class="project-meta">Updated ${escapeHtml(timeAgo(p.updated_at))}${p.license ? ` · ${escapeHtml(p.license)}` : ''}</div>
        <div class="project-actions">
          <button type="button" class="btn card-small-btn" data-act="open">Open</button>
          <button type="button" class="btn secondary card-small-btn" data-act="publish">Publish</button>
          <button type="button" class="btn secondary card-small-btn" data-act="rename">Rename</button>
          <button type="button" class="btn secondary card-small-btn" data-act="duplicate">Duplicate</button>
          <button type="button" class="btn secondary card-small-btn danger" data-act="delete">Delete</button>
        </div>
      </article>`).join('');
    const published = mine.analyses.map((a) => `
      <article class="project-card">
        <div class="project-top"><h3><a href="#/a/${a.id}">${escapeHtml(a.title)}</a></h3>${a.verified ? '<span class="chip lic-open">Verified</span>' : '<span class="chip lic-restricted">Unverified</span>'}</div>
        <p class="hint">${escapeHtml(a.visibility === 'public' ? 'Public' : 'Link only')} · ${a.views} views · ${a.forks} forks · ${escapeHtml(timeAgo(a.created_at))}</p>
        <div class="project-actions"><a class="btn secondary card-small-btn" href="#/a/${a.id}">View</a><button type="button" class="btn secondary card-small-btn" data-copy="${a.id}">Copy link</button></div>
      </article>`).join('');
    body.innerHTML = `
      <div class="projects-head"><span class="hint">${projects.length} of ${limit} projects</span></div>
      ${projects.length ? `<div class="project-grid">${cards}</div>` : `<div class="panel gate-panel"><h2>No projects yet</h2><p class="hint">Load a dataset, then click “Save analysis” to keep it here.</p><a class="btn" href="#/datasets">Find a dataset</a></div>`}
      <h2 class="section-title">Published analyses</h2>
      ${mine.analyses.length ? `<div class="project-grid">${published}</div>` : '<p class="hint">Nothing published yet. Use “Publish analysis” on the Export page to share a verified analysis.</p>'}`;
    body.querySelectorAll('.project-card[data-project]').forEach((card) => {
      const id = card.dataset.project;
      const project = projects.find((p) => p.id === id);
      card.querySelector('[data-act=open]').addEventListener('click', () => openProject(id));
      card.querySelector('[data-act=publish]').addEventListener('click', async () => {
        const opened = await openProject(id);
        if (opened) { navigate('export'); setTimeout(() => openPublishDialog(), 300); }
      });
      card.querySelector('[data-act=rename]').addEventListener('click', async () => {
        const name = prompt('Rename project', project.name);
        if (!name || !name.trim()) return;
        try {
          await api(`/api/projects/${id}`, {method: 'PUT', body: {name}});
          if (state.projectId === id) state.projectName = name.trim();
          PAGE_HOOKS.projects();
        } catch (error) { toast(error.message, 'bad'); }
      });
      card.querySelector('[data-act=duplicate]').addEventListener('click', async () => {
        try { await api(`/api/projects/${id}/duplicate`, {method: 'POST', body: {}}); toast('Project duplicated'); PAGE_HOOKS.projects(); } catch (error) { toast(error.message, 'bad'); }
      });
      card.querySelector('[data-act=delete]').addEventListener('click', async () => {
        if (!confirm(`Delete “${project.name}”? This cannot be undone.`)) return;
        try { await api(`/api/projects/${id}`, {method: 'DELETE'}); if (state.projectId === id) { state.projectId = null; state.projectName = null; } toast('Project deleted'); PAGE_HOOKS.projects(); } catch (error) { toast(error.message, 'bad'); }
      });
    });
    body.querySelectorAll('[data-copy]').forEach((btn) => btn.addEventListener('click', () => copyLink(btn.dataset.copy)));
  } catch (error) {
    body.innerHTML = `<p class="hint">${escapeHtml(error.message)}</p>`;
  }
};

// The datasets page also shows your latest cloud projects (or invites you to sign in).
const previousDatasetsHook = PAGE_HOOKS.datasets;
PAGE_HOOKS.datasets = async function renderDatasetsExtras() {
  if (previousDatasetsHook) previousDatasetsHook();
  let host = $('cloud-preview');
  if (!host) {
    host = document.createElement('div');
    host.id = 'cloud-preview';
    $('past-analyses-panel').insertBefore(host, $('past-analyses-list'));
  }
  if (!state.user) {
    host.innerHTML = '<div class="cloud-invite"><strong>Keep your work everywhere.</strong> Sign in to save projects to your account instead of just this browser. <button type="button" class="link-btn" data-signin>Sign in</button></div>';
    host.querySelector('[data-signin]').addEventListener('click', () => openAuth({reason: 'Sign in to save analyses to your account.'}));
    return;
  }
  try {
    const {projects} = await api('/api/projects');
    host.innerHTML = projects.length
      ? `<h3 class="mini-title">Your projects</h3><div class="mini-projects">${projects.slice(0, 4).map((p) => `<button type="button" class="mini-project" data-open="${p.id}"><strong>${escapeHtml(p.name)}</strong><span>${escapeHtml(p.dataset_name)} · ${escapeHtml(timeAgo(p.updated_at))}</span></button>`).join('')}</div><a class="link-btn" href="#/projects">All projects →</a><h3 class="mini-title">In this browser</h3>`
      : '';
    host.querySelectorAll('[data-open]').forEach((b) => b.addEventListener('click', () => openProject(b.dataset.open)));
  } catch (error) { host.innerHTML = ''; }
};

// ================================================================= publish + explore + analysis page

const publishState = {mode: 'create', share: null};

function updatePublishAvailability() {
  const btn = $('publish-open-btn');
  if (!btn) return;
  $('publish-status').textContent = state.user ? '' : 'You will be asked to sign in.';
}

function reuseAllowedForCurrentDataset() {
  const info = state.dataset?.license_info;
  const isKaggle = Boolean(state.slug);
  if (isKaggle && info && !['open', 'conditions'].includes(info.level)) return {ok: false, note: `Disabled: this dataset's license (${info.name || 'not specified'}) does not clearly allow redistribution.`};
  if (isKaggle && !info) return {ok: false, note: 'Disabled: the dataset license could not be confirmed.'};
  return {ok: true, note: isKaggle ? `Allowed by the dataset's license (${info.name}). Credit the source.` : 'Only enable this for data you have the right to share.'};
}

function openPublishDialog({mode = 'create', share = null} = {}) {
  if (mode === 'create' && !state.analysis) { toast('Load a dataset first.', 'bad'); return; }
  if (!state.user) { openAuth({reason: 'Sign in to publish a verified analysis.', onDone: () => openPublishDialog({mode, share})}); return; }
  publishState.mode = mode;
  publishState.share = share;
  showError($('publish-error'), '');
  const creating = mode === 'create';
  $('publish-title').textContent = creating ? 'Publish a verified analysis' : 'Edit published analysis';
  $('publish-confirm').textContent = creating ? 'Publish and verify' : 'Save changes';
  $('publish-name').value = creating ? (exportPrefs.pdf.title || `${state.analysis.dataset_name} analysis`) : share.title;
  $('publish-description').value = creating ? '' : share.description;
  $('publish-visibility').querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.value === (creating ? 'public' : share.visibility)));
  const reuseRow = $('publish-reuse').closest('label');
  reuseRow.hidden = !creating;
  $('publish-reuse-note').hidden = !creating;
  if (creating) {
    const policy = reuseAllowedForCurrentDataset();
    $('publish-reuse').checked = false;
    $('publish-reuse').disabled = !policy.ok;
    $('publish-reuse-note').textContent = policy.note;
  }
  openOverlay('publish-overlay');
  $('publish-name').focus();
}

bindOverlayClose('publish-overlay', ['publish-close', 'publish-cancel']);
bindSeg($('publish-visibility'));
$('publish-open-btn').addEventListener('click', () => openPublishDialog());
$('publish-confirm').addEventListener('click', async () => {
  const title = $('publish-name').value.trim();
  if (!title) { showError($('publish-error'), 'A title is required.'); return; }
  const button = $('publish-confirm');
  button.disabled = true;
  try {
    if (publishState.mode === 'edit') {
      await api(`/api/shares/${publishState.share.id}`, {method: 'PUT', body: {title, description: $('publish-description').value, visibility: segValue($('publish-visibility'))}});
      closeOverlay('publish-overlay');
      toast('Changes saved');
      PAGE_HOOKS.analysis(publishState.share.id);
      return;
    }
    button.textContent = 'Verifying...';
    const p = projectPayload();
    const result = await api('/api/shares', {method: 'POST', body: {...p, title, description: $('publish-description').value, visibility: segValue($('publish-visibility')), allow_data_reuse: $('publish-reuse').checked}});
    closeOverlay('publish-overlay');
    toast(result.verified ? 'Published and verified ✓' : 'Published, but verification found a problem', result.verified ? 'ok' : 'bad');
    navigate(`a/${result.id}`);
  } catch (error) {
    showError($('publish-error'), error.message);
  } finally {
    button.disabled = false;
    button.textContent = publishState.mode === 'edit' ? 'Save changes' : 'Publish and verify';
  }
});

function copyLink(id) {
  const url = `${window.location.origin}/#/a/${id}`;
  (navigator.clipboard ? navigator.clipboard.writeText(url) : Promise.reject()).then(() => toast('Link copied'), () => { prompt('Copy this link', url); });
}

// ---- Explore
const explore = {q: '', sort: 'new', verified: false, timer: null, bound: false};

async function loadExplore() {
  const grid = $('explore-grid');
  try {
    const params = new URLSearchParams({sort: explore.sort});
    if (explore.q) params.set('q', explore.q);
    if (explore.verified) params.set('verified', '1');
    const [{analyses, total}, stats] = await Promise.all([api(`/api/shares?${params}`), api('/api/graph/stats').catch(() => null)]);
    $('explore-sub').textContent = `${total} published analys${total === 1 ? 'is' : 'es'}${stats ? ` · ${stats.indexed} datasets indexed in the dataset graph` : ''}. Every one can be re-verified from its stored data.`;
    grid.innerHTML = analyses.length ? analyses.map((a) => `
      <a class="explore-card" href="#/a/${a.id}">
        <div class="explore-top">${a.verified ? '<span class="chip lic-open">✓ Verified</span>' : '<span class="chip lic-restricted">Unverified</span>'}<span class="hint">${escapeHtml(timeAgo(a.created_at))}</span></div>
        <h3>${escapeHtml(a.title)}</h3>
        <p class="explore-by">by ${escapeHtml(a.author)} · ${escapeHtml(a.dataset_title || '')}</p>
        <p class="explore-desc">${escapeHtml(a.description || a.top_finding || '')}</p>
        ${a.top_finding ? `<p class="explore-finding">${escapeHtml(a.top_finding)}</p>` : ''}
        <div class="explore-foot"><span>${(a.rows || 0).toLocaleString()} rows</span><span>Quality ${a.quality_score ?? '-'}/100</span><span>⑂ ${a.forks}</span><span>👁 ${a.views}</span></div>
      </a>`).join('') : '<div class="panel gate-panel"><h2>Nothing here yet</h2><p class="hint">Be the first: analyze a dataset and publish it from the Export page.</p></div>';
  } catch (error) {
    grid.innerHTML = `<p class="hint">${escapeHtml(error.message)}</p>`;
  }
}

PAGE_HOOKS.explore = function renderExplorePage() {
  if (!explore.bound) {
    explore.bound = true;
    $('explore-search').addEventListener('input', (event) => { explore.q = event.target.value.trim(); clearTimeout(explore.timer); explore.timer = setTimeout(loadExplore, 250); });
    bindSeg($('explore-sort'), (value) => { explore.sort = value; loadExplore(); });
    $('explore-verified').addEventListener('change', (event) => { explore.verified = event.target.checked; loadExplore(); });
  }
  loadExplore();
};

// ---- A published analysis
function ratingChip(rating) {
  const cls = {High: 'lic-open', Moderate: 'lic-conditions', Low: 'lic-restricted', 'Null result': 'lic-unknown'}[rating] || 'lic-unknown';
  return `<span class="chip ${cls}">${escapeHtml(rating)}</span>`;
}

function renderVerificationPanel(a) {
  const v = a.verification;
  if (!v) return '<div class="verify-panel verify-none"><strong>Not yet verified</strong></div>';
  const ok = v.verified;
  return `
    <div class="verify-panel ${ok ? 'verify-ok' : 'verify-bad'}" id="verify-panel">
      <div class="verify-icon">${ok ? '✓' : '!'}</div>
      <div class="verify-text">
        <strong>${ok ? 'Verified' : 'Verification failed'}</strong>
        <span>${v.error ? escapeHtml(v.error) : `${v.passed} of ${v.checked} checks reproduced from the stored data${v.ran_at ? ` · last run ${escapeHtml(new Date(v.ran_at).toLocaleString())}` : ''}`}</span>
        <small>Engine ${escapeHtml(v.engine || '')} · ${escapeHtml(v.software || '')}</small>
      </div>
      <div class="verify-actions">
        <button type="button" class="btn secondary card-small-btn" id="verify-again">Re-run verification</button>
        <button type="button" class="btn secondary card-small-btn" id="verify-details">See checks</button>
      </div>
    </div>
    <div id="verify-checks" class="verify-checks" hidden></div>`;
}

PAGE_HOOKS.analysis = async function renderAnalysisPage(id) {
  const body = $('analysis-body');
  if (!id) return;
  body.innerHTML = '<p class="hint">Loading analysis...</p>';
  let a;
  try {
    a = (await api(`/api/shares/${id}`)).analysis;
  } catch (error) {
    body.innerHTML = `<div class="panel gate-panel"><h2>Analysis not found</h2><p class="hint">${escapeHtml(error.message)}</p><a class="btn" href="#/explore">Browse analyses</a></div>`;
    return;
  }
  document.title = `${a.title} · DataScopia`;
  const ds = a.dataset;
  const lic = ds.license_info;
  const canFork = a.allow_data_reuse;
  body.innerHTML = `
    <div class="analysis-hero">
      <div class="explore-top"><span class="chip">${escapeHtml(a.visibility === 'public' ? 'Public' : 'Link only')}</span><span class="hint">Published ${escapeHtml(timeAgo(a.created_at))} by <strong>${escapeHtml(a.author)}</strong></span></div>
      <h1>${escapeHtml(a.title)}</h1>
      ${a.description ? `<p class="analysis-desc">${escapeHtml(a.description)}</p>` : ''}
      <div class="analysis-actions">
        <button type="button" class="btn" id="fork-btn" ${canFork ? '' : 'disabled'} title="${canFork ? '' : escapeHtml(a.data_note || 'Forking is not allowed for this analysis.')}">Fork to my projects</button>
        <a class="btn secondary" href="${API_BASE}/api/shares/${a.id}/report.pdf">Download PDF</a>
        <a class="btn secondary" href="${API_BASE}/api/shares/${a.id}/notebook.ipynb">Download notebook</a>
        <button type="button" class="btn secondary" id="copy-link-btn">Copy link</button>
        ${a.is_owner ? '<button type="button" class="btn secondary" id="edit-share-btn">Edit</button><button type="button" class="btn secondary danger" id="unpublish-btn">Unpublish</button>' : ''}
      </div>
      ${canFork ? '' : `<p class="ef-hint">${escapeHtml(a.data_note || '')}</p>`}
    </div>
    ${renderVerificationPanel(a)}
    <div class="analysis-grid">
      <section class="panel"><h3>Reproducibility</h3>
        <p class="hint">This hash fingerprints the original data, the cleaning steps and every reported number. Equal hashes mean an identical analysis.</p>
        <div class="hash-row"><code>${escapeHtml(a.repro_hash)}</code><button type="button" class="link-btn" id="copy-hash">Copy</button></div>
        <div class="analysis-tiles">
          <div><strong>${a.headline.rows.toLocaleString()}</strong><span>rows analysed</span></div>
          <div><strong>${a.headline.columns}</strong><span>columns</span></div>
          <div><strong>${a.headline.quality_score}/100</strong><span>data quality</span></div>
          <div><strong>${a.steps.length}</strong><span>cleaning steps</span></div>
        </div>
      </section>
      <section class="panel"><h3>Data source</h3>
        <p><strong>${escapeHtml(ds.title)}</strong></p>
        <p class="hint">${escapeHtml(ds.source === 'kaggle' ? `Kaggle dataset ${ds.slug}` : ds.source === 'join' ? 'Combined dataset (dataset graph)' : ds.source === 'library' ? 'Built-in library' : 'Uploaded file')} · ${ds.rows.toLocaleString()} rows × ${ds.cols} columns${ds.sample && ds.sample.is_sample ? ' (partial sample)' : ''}</p>
        ${ds.license ? `<p>License: <strong>${escapeHtml(ds.license)}</strong> ${lic ? ratingLicenseChip(lic) : ''}</p>` : '<p class="hint">No license recorded.</p>'}
        ${ds.url ? `<a href="${escapeHtml(ds.url)}" target="_blank" rel="noreferrer">View the source</a>` : ''}
      </section>
    </div>
    <section class="panel"><h3>Evidence</h3>
      <p class="hint">Each headline finding, what it was checked against, and how far it can be trusted.</p>
      ${a.evidence.length ? `<div class="table-wrap"><table class="evidence-table"><thead><tr><th>Finding</th><th>Statistic</th><th>Robustness</th><th>Rating</th></tr></thead><tbody>${a.evidence.map((e) => `<tr><td>${escapeHtml(e.finding)}</td><td>${escapeHtml(e.statistic)}</td><td>${escapeHtml(e.robustness)}</td><td>${ratingChip(e.rating)}</td></tr>`).join('')}</tbody></table></div>` : '<p class="hint">No headline findings met the reporting thresholds.</p>'}
    </section>
    <section class="panel"><h3>Charts</h3><div class="chart-grid" id="analysis-charts">${a.charts.map((c, i) => `<div class="chart-card"><div class="card-header"><span class="card-title"><span class="type-dot"></span><span>${escapeHtml(c.customTitle || c.title || c.type)}</span></span></div><div class="chart-area" id="analysis-chart-${i}"></div><div class="chart-notes" id="analysis-chart-notes-${i}" hidden></div></div>`).join('') || '<p class="hint">No charts.</p>'}</div></section>
    ${a.steps.length ? `<section class="panel"><h3>Cleaning steps applied</h3><ol class="steps-list">${a.steps.map((s) => `<li>${escapeHtml(s.description || s.type)}</li>`).join('')}</ol></section>` : ''}
    ${a.findings.length ? `<section class="panel"><h3>Findings from the analyst</h3><ul class="steps-list">${a.findings.map((f) => `<li><strong>${escapeHtml(f.title || 'Finding')}</strong> ${escapeHtml(f.type === 'question' ? f.answer : (f.summary || ''))}</li>`).join('')}</ul></section>` : ''}`;

  // actions
  $('copy-link-btn').addEventListener('click', () => copyLink(a.id));
  $('copy-hash').addEventListener('click', () => { navigator.clipboard?.writeText(a.repro_hash); toast('Hash copied'); });
  $('fork-btn').addEventListener('click', () => requireUser('Sign in to fork this analysis into your own projects.', async () => {
    try {
      const {project_id: projectId} = await api(`/api/shares/${a.id}/fork`, {method: 'POST', body: {}});
      toast('Forked to your projects');
      await openProject(projectId);
    } catch (error) { toast(error.message, 'bad'); }
  }));
  wireVerification(a);
  if (a.is_owner) {
    $('edit-share-btn').addEventListener('click', () => openPublishDialog({mode: 'edit', share: a}));
    $('unpublish-btn').addEventListener('click', async () => {
      if (!confirm('Unpublish this analysis? Its link will stop working and its stored data will be deleted.')) return;
      try { await api(`/api/shares/${a.id}`, {method: 'DELETE'}); toast('Unpublished'); navigate('projects'); } catch (error) { toast(error.message, 'bad'); }
    });
  }
  // charts load one at a time so the page appears immediately
  for (let i = 0; i < a.charts.length; i += 1) {
    const holder = $(`analysis-chart-${i}`);
    if (!holder) break;
    try {
      const result = await api(`/api/shares/${a.id}/chart/${i}`);
      if (routeParam !== a.id) return;
      const meta = result.meta || {};
      Plotly.react(holder, result.data || [], {...PLOTLY_LAYOUT_BASE, ...(result.layout || {}), title: {text: meta.title || '', x: 0.02, xanchor: 'left', font: {size: 14}}}, PLOTLY_CONFIG);
      const notes = [...(meta.notes || [])];
      const notesEl = $(`analysis-chart-notes-${i}`);
      notesEl.hidden = notes.length === 0;
      notesEl.innerHTML = notes.map((n) => `<p>${escapeHtml(n)}</p>`).join('');
    } catch (error) {
      holder.innerHTML = `<p class="hint" style="padding:16px">${escapeHtml(error.message)}</p>`;
    }
  }
};

function ratingLicenseChip(info) {
  const cls = {open: 'lic-open', conditions: 'lic-conditions', restricted: 'lic-restricted', unknown: 'lic-unknown'}[info.level] || 'lic-unknown';
  return `<span class="chip ${cls}">${escapeHtml({open: 'Open', conditions: 'Conditions apply', restricted: 'Restricted', unknown: 'Unknown'}[info.level] || '')}</span>`;
}

function wireVerification(a) {
  const again = $('verify-again');
  const details = $('verify-details');
  if (again) {
    again.addEventListener('click', async () => {
      again.disabled = true;
      again.textContent = 'Re-running...';
      try {
        const {verification} = await api(`/api/shares/${a.id}/verify`, {method: 'POST', body: {}});
        a.verification = verification;
        $('verify-panel').outerHTML = renderVerificationPanel(a).split('<div id="verify-checks"')[0];
        toast(verification.verified ? 'Verified again ✓' : 'Verification failed', verification.verified ? 'ok' : 'bad');
        wireVerification(a);
        fillChecks(verification);
      } catch (error) {
        toast(error.message, 'bad');
        again.disabled = false;
        again.textContent = 'Re-run verification';
      }
    });
  }
  if (details) {
    details.addEventListener('click', async () => {
      const host = $('verify-checks');
      host.hidden = !host.hidden;
      if (!host.hidden && !host.dataset.filled) {
        try { fillChecks((await api(`/api/shares/${a.id}/verification`)).verification); } catch (error) { host.textContent = error.message; }
      }
    });
  }
}

function fillChecks(v) {
  const host = $('verify-checks');
  if (!host || !v) return;
  host.dataset.filled = '1';
  const fmt = (x) => (x === null || x === undefined ? '–' : typeof x === 'number' ? String(Math.round(x * 1e6) / 1e6) : String(x));
  host.innerHTML = `<div class="table-wrap"><table class="evidence-table"><thead><tr><th>Check</th><th>Published</th><th>Recomputed</th><th></th></tr></thead><tbody>${(v.checks || []).map((c) => `<tr class="${c.ok ? '' : 'check-bad'}"><td>${escapeHtml(c.name)}</td><td><code>${escapeHtml(fmt(c.expected))}</code></td><td><code>${escapeHtml(fmt(c.got))}</code></td><td>${c.ok ? '✓' : '✗'}</td></tr>`).join('')}</tbody></table></div>`;
}

// ================================================================= dataset graph

const graph = {scannedFor: null, running: false, joinTarget: null, how: 'left'};

async function scanGraph(force = false) {
  if (!state.analysis || !state.dataset?.dataset_id) return;
  const key = `${state.dataset.dataset_id}|${state.transformLog.length}`;
  if (!force && graph.scannedFor === key) return;
  if (graph.running) return;
  graph.running = true;
  graph.scannedFor = key;
  const results = $('graph-results');
  results.innerHTML = '<p class="hint">Scanning for datasets that can be combined...</p>';
  renderProvenance();
  try {
    const data = await api('/api/graph/joinable', {method: 'POST', body: {dataset_id: state.dataset.dataset_id, dataset_name: state.analysis.dataset_name}});
    const labels = {country: 'Country', us_state: 'US state', year: 'Year', entity: 'Entity', identifier: 'Identifier'};
    $('graph-keys').innerHTML = data.keys.length
      ? `<span class="hint">Join keys detected:</span> ${data.keys.map((k) => `<span class="chip">${escapeHtml(labels[k.kind] || k.kind)}: ${escapeHtml(k.column)}</span>`).join(' ')}`
      : '';
    $('graph-stat').innerHTML = `<strong>${data.indexed}</strong><span>datasets indexed</span>`;
    $('graph-rescan').hidden = false;
    if (!data.keys.length) {
      results.innerHTML = '<p class="hint">No join keys (country, US state, year, ID) were detected in this dataset, so there is nothing to match against. Add a country or ID column and scan again.</p>';
      return;
    }
    if (!data.candidates.length) {
      results.innerHTML = `<p class="hint">No dataset in the catalogue shares these keys yet. The catalogue grows every time someone analyses a Kaggle dataset${state.user ? '' : ', and your own saved projects appear here once you sign in'}.</p>`;
      return;
    }
    const badge = {kaggle: 'Kaggle', library: 'Built-in', project: 'Your project'};
    results.innerHTML = data.candidates.map((c, i) => `
      <article class="graph-card">
        <div class="graph-card-top"><span class="chip ${c.best.strength === 'high' ? 'lic-open' : c.best.strength === 'medium' ? 'lic-conditions' : 'lic-restricted'}">${escapeHtml(c.best.strength)} match</span><span class="chip">${escapeHtml(badge[c.type] || c.type)}</span></div>
        <h4>${escapeHtml(c.title)}</h4>
        <p class="hint">${escapeHtml(c.best.description)}</p>
        <p class="graph-adds">Adds <strong>${c.adds_columns}</strong> column${c.adds_columns === 1 ? '' : 's'}${c.adds_measures.length ? `: ${escapeHtml(c.adds_measures.slice(0, 4).join(', '))}${c.adds_measures.length > 4 ? '…' : ''}` : ''}</p>
        ${c.license ? `<p class="hint">License: ${escapeHtml(c.license)}</p>` : ''}
        <div class="project-actions"><button type="button" class="btn card-small-btn" data-join="${i}">Preview &amp; combine</button>${c.url ? `<a class="btn secondary card-small-btn" href="${escapeHtml(c.url)}" target="_blank" rel="noreferrer">View source</a>` : ''}</div>
      </article>`).join('');
    results.querySelectorAll('[data-join]').forEach((btn) => btn.addEventListener('click', () => openJoin(data.candidates[Number(btn.dataset.join)])));
  } catch (error) {
    results.innerHTML = `<p class="hint">${escapeHtml(error.message)}</p>`;
    graph.scannedFor = null;
  } finally {
    graph.running = false;
  }
}

function renderProvenance() {
  const host = $('graph-hint');
  const d = state.dataset?.derived_from;
  if (d) {
    host.innerHTML = `<strong>This dataset was created by combining</strong> “${escapeHtml(d.left)}” with “${escapeHtml(d.right)}” (${d.stats.matched_left_pct}% of rows matched, ${d.stats.added_columns.length} columns added). You can keep combining it with more datasets below.`;
  } else {
    host.textContent = 'The dataset graph compares the join keys of this dataset (countries, years, IDs) with a growing catalogue, the built-in library and your own projects.';
  }
}

PAGE_HOOKS.overview = function overviewGraphHook() { scanGraph(false); };
$('graph-rescan').addEventListener('click', () => scanGraph(true));

function joinRequest() {
  const option = graph.joinTarget.options[Number($('join-option').value)];
  return {dataset_id: state.dataset.dataset_id, dataset_name: state.analysis.dataset_name, candidate_id: graph.joinTarget.id, option, how: graph.how};
}

async function refreshJoinPreview() {
  const box = $('join-preview');
  box.innerHTML = '<p class="hint">Building preview...</p>';
  $('join-confirm').disabled = true;
  showError($('join-error'), '');
  try {
    const p = await api('/api/graph/preview', {method: 'POST', body: joinRequest()});
    const s = p.stats;
    const cols = p.columns.slice(0, 8);
    box.innerHTML = `
      <div class="join-stats">
        <div><strong>${s.matched_left_pct}%</strong><span>of your rows matched</span></div>
        <div><strong>${s.result_rows.toLocaleString()}</strong><span>rows in result</span></div>
        <div><strong>${s.added_columns.length}</strong><span>columns added</span></div>
      </div>
      ${s.warnings.map((w) => `<p class="join-warning">⚠ ${escapeHtml(w)}</p>`).join('')}
      <div class="table-wrap"><table class="evidence-table"><thead><tr>${cols.map((c) => `<th>${escapeHtml(c)}</th>`).join('')}</tr></thead><tbody>${p.rows.map((r) => `<tr>${cols.map((c) => `<td>${escapeHtml(r[c] === null || r[c] === undefined ? '' : r[c])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
      <p class="hint">Added: ${escapeHtml(s.added_columns.slice(0, 10).join(', '))}${s.added_columns.length > 10 ? '…' : ''}</p>
      ${p.candidate_license ? `<p class="hint">The other dataset's license: ${escapeHtml(p.candidate_license)}. Combined data is subject to both licenses.</p>` : ''}`;
    $('join-confirm').disabled = false;
  } catch (error) {
    box.innerHTML = '';
    showError($('join-error'), error.message);
  }
}

function openJoin(candidate) {
  graph.joinTarget = candidate;
  graph.how = 'left';
  $('join-title').textContent = `Combine with: ${candidate.title}`;
  $('join-sub').textContent = 'Your dataset stays as it is; a new combined dataset is created and opened.';
  $('join-option').innerHTML = candidate.options.map((o, i) => `<option value="${i}">${escapeHtml(o.kind.replace('+', ' + '))}: ${escapeHtml(o.description)}</option>`).join('');
  $('join-how').querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.value === 'left'));
  openOverlay('join-overlay');
  refreshJoinPreview();
}

bindOverlayClose('join-overlay', ['join-close', 'join-cancel']);
bindSeg($('join-how'), (value) => { graph.how = value; refreshJoinPreview(); });
$('join-option').addEventListener('change', refreshJoinPreview);
$('join-confirm').addEventListener('click', async () => {
  const button = $('join-confirm');
  button.disabled = true;
  button.textContent = 'Combining...';
  try {
    const payload = await api('/api/graph/join', {method: 'POST', body: joinRequest()});
    closeOverlay('join-overlay');
    applyWorkspacePayload(payload, {findings: [], projectId: null});
    toast(`Created “${payload.dataset.name}” with ${payload.provenance.stats.added_columns.length} new columns`);
    navigate('overview');
    scanGraph(true);
  } catch (error) {
    showError($('join-error'), error.message);
  } finally {
    button.disabled = false;
    button.textContent = 'Create combined dataset';
  }
});

// ================================================================= start-up

refreshUser().then(() => renderRoute());
