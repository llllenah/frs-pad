// Односторінковий інтерфейс прототипу: форми 4.7 (сеанс верифікації) і 4.8 (запис сеансів),
// вкладка «Моніторинг» (ідентифікація 1:N). Браузер захоплює кадри камери, надсилає їх серверу
// через WebSocket /ws/stream (8 байт часу кадру + JPEG) і малює відповідь: точки обличчя і кисті,
// стан сеансу, ряди s_ref(t), s_enr(t), ознаки, оцінку і рішення або рамки розпізнаних облич.
'use strict';

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

// Число з десятковою комою, як у тексті записки.
const fmt = (x, d = 2) => (x == null || !Number.isFinite(x)) ? '–' : x.toFixed(d).replace('.', ',');
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const lowerFirst = s => s ? s[0].toLowerCase() + s.slice(1) : s;
const upperFirst = s => s ? s[0].toUpperCase() + s.slice(1) : s;

const FINAL = new Set(['bona_fide', 'attack', 'failed', 'cancelled']);
const isActive = s => s && !FINAL.has(s.state);
const DECISION_TEXT = {
  bona_fide: 'СПРАВЖНЄ ПОДАННЯ', attack: 'АТАКА ПІДМІНИ ОБЛИЧЧЯ', failed: 'НЕ ПРОЙДЕНО', cancelled: 'СЕАНС ПЕРЕРВАНО',
};
const STORED_TEXT = {
  encrypted: 'зашифрований файл', plain: 'відкритий .npy (застарілий)', memory: 'лише в пам\'яті сервера',
  locked: 'зашифровано, ключ не задано',
};
const MAIN_FEATURES = [
  ['s_ref_before_mean', 'Подібність до виклику', 2],
  ['s_ref_during_min', 'Мін. під час виклику', 2],
  ['s_ref_during_jump', 'Макс. міжкадровий стрибок', 2],
  ['s_ref_during_var', 'Дисперсія (під час)', 3],
  ['s_ref_trec', 'Час відновлення', 1, ' с'],
];
const SUBTEXT = {
  cover: 'Утримуйте положення до завершення відліку, потім опустіть руку',
  turn_head: 'Утримуйте поворот, потім поверніться обличчям до камери',
  hand_swipe: 'Проведіть кистю перед обличчям з одного боку на інший',
};
const SIDE_TEXT = { left: 'ліва', right: 'права', center: 'центр' };
const LIGHT_SHORT = { normal: 'норм.', low: 'слабке', bright: 'яскраве', backlight: 'контрове' };
// Рамки моніторингу: відома особа – зелена, невідома – червона (єдині кольорові елементи інтерфейсу).
const KNOWN_COLOR = '#16a34a', UNKNOWN_COLOR = '#dc2626';
const FAIL_REASON = {
  timeout: 'перевищено тривалість сеансу', no_face: 'обличчя не виявлено', not_performed: 'виклики не виконано',
  error: 'збій модуля',
};
const LEVEL_TEXT = { alarm: 'тривога', warn: 'не пройдено', ok: 'підтверджено', info: 'у базі' };

const App = {
  config: null, participants: [], view: 'verify',
  session: null, frame: null, enroll: null, recordings: null, monitor: null,
  incidents: [], lastEventId: null, lastHash: '#/verify',
};

// ----- звукові сповіщення (WebAudio, без зовнішніх файлів) -----

const Sound = {
  ctx: null, on: true,
  // тривога – чотири різкі сигнали, «не пройдено» – два нижчі, справжнє подання – тихий дзвін
  PATTERNS: {
    alarm: [[880, 0, 0.16, 'square'], [660, 0.2, 0.16, 'square'], [880, 0.4, 0.16, 'square'], [660, 0.6, 0.16, 'square']],
    warn: [[520, 0, 0.22, 'triangle'], [415, 0.3, 0.3, 'triangle']],
    ok: [[660, 0, 0.14, 'sine'], [990, 0.15, 0.3, 'sine']],
  },

  init() {
    try { this.on = localStorage.getItem('frs-sound') !== 'off'; } catch (_) { /* сховище недоступне */ }
    // браузер дозволяє звук лише після дії користувача, тому контекст створюється на першому кліку
    const unlock = () => this.context();
    document.addEventListener('pointerdown', unlock, { once: true });
    document.addEventListener('keydown', unlock, { once: true });
    this.button();
  },
  context() {
    if (!this.ctx) {
      try { this.ctx = new (window.AudioContext || window.webkitAudioContext)(); } catch (_) { return null; }
    }
    if (this.ctx.state === 'suspended') this.ctx.resume();
    return this.ctx;
  },
  toggle() {
    this.on = !this.on;
    try { localStorage.setItem('frs-sound', this.on ? 'on' : 'off'); } catch (_) { /* лише на цей сеанс */ }
    this.button();
    if (this.on) this.play('ok');
  },
  button() {
    const b = $('#sound');
    b.textContent = this.on ? 'Звук: увімк.' : 'Звук: вимк.';
    b.classList.toggle('off', !this.on);
    b.setAttribute('aria-pressed', String(this.on));
  },
  play(kind) {
    if (!this.on) return;
    const ctx = this.context();
    if (!ctx) return;
    const t0 = ctx.currentTime + 0.02;
    const peak = kind === 'alarm' ? 0.08 : 0.12;
    for (const [f, at, dur, type] of this.PATTERNS[kind] || []) {
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.type = type;
      o.frequency.value = f;
      g.gain.setValueAtTime(0.0001, t0 + at);
      g.gain.exponentialRampToValueAtTime(peak, t0 + at + 0.015);
      g.gain.exponentialRampToValueAtTime(0.0001, t0 + at + dur);
      o.connect(g).connect(ctx.destination);
      o.start(t0 + at);
      o.stop(t0 + at + dur + 0.03);
    }
  },
};

// ----- журнал інцидентів і сповіщення охоронця -----

// Текст події для журналу, сповіщення і вікна подробиць.
function incidentView(e) {
  if (e.kind === 'unknown') {
    return { title: 'Невідома особа', head: 'ТРИВОГА · НЕВІДОМА ОСОБА',
      sub: e.similarity != null ? `подібність ${fmt(e.similarity)} – обличчя немає в базі` : 'база еталонів порожня' };
  }
  if (e.kind === 'known') return { title: `${e.person_id} – особа з бази`, head: '', sub: `подібність ${fmt(e.similarity)}` };
  const who = e.person_id;
  if (e.decision === 'attack') {
    return { title: `${who} – атака підміни обличчя`, head: 'ТРИВОГА · АТАКА ПІДМІНИ ОБЛИЧЧЯ',
      sub: `обличчя схоже на ${who}, але перевірка викликами виявила підміну · оцінка ${fmt(e.score)}` };
  }
  if (e.decision === 'failed') {
    return { title: `${who} – перевірку не пройдено`, head: 'ПЕРЕВІРКУ НЕ ПРОЙДЕНО',
      sub: `${FAIL_REASON[e.reason] || 'виклики не виконано'} · особу не підтверджено` };
  }
  return { title: `${who} – справжнє подання`, head: 'ПЕРЕВІРКУ ПРОЙДЕНО',
    sub: `виклики виконано, ознак підміни немає · оцінка ${fmt(e.score)}` };
}

const snapshotUrl = e => `/api/incidents/${e.id}/snapshot.jpg`;
// Знімок міг зникнути з пам'яті сервера (перезапуск, понад 50 знімків) – тоді показується порожня рамка.
const thumb = (e, cls = '') => e.snapshot
  ? `<img class="${cls}" src="${snapshotUrl(e)}" alt="" onerror="this.outerHTML='<span class=&quot;noimg&quot;></span>'">`
  : `<span class="noimg ${cls}"></span>`;

// Нові події з сервера: оновлює журнал і показує сповіщення лише для подій, яких ще не було.
// Повтори не надходять: сервер пише подію, коли особу підтверджено кілька кадрів поспіль,
// і повторює її лише після перерви (журнал моніторингу з уже наявним усуненням дублікатів).
function ingestEvents(events, silent = false) {
  if (!events) return;
  const byId = new Map(App.incidents.map(e => [e.id, e]));
  for (const e of events) byId.set(e.id, e);
  App.incidents = [...byId.values()].sort((a, b) => b.id - a.id).slice(0, 100);
  const maxId = App.incidents.length ? App.incidents[0].id : 0;
  if (App.lastEventId == null || silent) {
    App.lastEventId = maxId;
    return;
  }
  const fresh = events.filter(e => e.id > App.lastEventId).sort((a, b) => a.id - b.id);
  App.lastEventId = Math.max(App.lastEventId, maxId);
  for (const e of fresh) if (e.level !== 'info') Popups.show(e);
}

async function refreshIncidents(silent = false) {
  try { ingestEvents(await api('/api/incidents'), silent); } catch (_) { /* сервер недоступний */ }
  render();
}

const Popups = {
  show(e) {
    const v = incidentView(e);
    const box = $('#popups');
    const el = document.createElement('div');
    el.className = `popup ${e.level}`;
    el.setAttribute('role', e.level === 'ok' ? 'status' : 'alert');
    const action = e.review
      ? `<button class="btn sm" data-act="review">Переглянути сеанс</button>`
      : `<button class="btn sm" data-act="details">Деталі</button>`;
    el.innerHTML = `${thumb(e)}
      <div class="txt"><b>${esc(v.head)}</b><span>${esc(e.kind === 'unknown' ? 'Обличчя в кадрі не знайдено в базі еталонів' : v.title)}</span>
        <small>${esc(e.time)} · ${esc(v.sub)}</small></div>
      <div class="acts">${action}<button class="btn sm" data-act="close">${e.level === 'ok' ? 'Закрити' : 'Прийнято'}</button></div>`;
    el.addEventListener('click', ev => {
      const act = ev.target.closest('[data-act]');
      if (!act) return;
      el.remove();
      if (act.dataset.act === 'review') location.hash = `#/review/${encodeURIComponent(e.review)}`;
      else if (act.dataset.act === 'details') openIncident(e.id);
    });
    box.prepend(el);
    while (box.children.length > 3) box.lastElementChild.remove();
    if (e.level === 'ok') setTimeout(() => el.remove(), 8000);
    Sound.play(e.level);
  },
};

function openIncident(id) {
  const e = App.incidents.find(x => x.id === id);
  if (!e) return;
  const v = incidentView(e);
  $('#inc-title').textContent = v.title;
  $('#inc-photo').innerHTML = e.snapshot ? `<img src="${snapshotUrl(e)}" alt="Знімок обличчя">`
    : '<div class="noimg">знімка немає<br><small>зберігаються лише останні 50 і лише до перезапуску сервера</small></div>';
  const rows = [['Час', `${e.date || ''} ${e.time}`.trim()]];
  if (e.kind === 'pad') {
    rows.push(['Подія', 'перевірка активними викликами'], ['Заявлена особа', e.person_id],
      ['Рішення', DECISION_TEXT[e.decision] ? DECISION_TEXT[e.decision].toLowerCase() : e.decision],
      ['Оцінка атаки', fmt(e.score)], ['Сеанс', e.code], ['Запущено з', e.origin === 'monitor' ? 'моніторингу' : 'вкладки «Верифікація»']);
    if (e.decision === 'failed') rows.push(['Причина', FAIL_REASON[e.reason] || '–']);
  } else {
    rows.push(['Подія', e.kind === 'unknown' ? 'невідома особа в кадрі' : 'відома особа в кадрі'],
      ['Особа', e.kind === 'unknown' ? 'невідома' : e.person_id], ['Подібність', fmt(e.similarity)]);
    if (e.kind === 'unknown' && e.nearest) rows.push(['Найближчий еталон', e.nearest]);
  }
  $('#inc-info').innerHTML = rows.map(([k, x]) => `<dt>${esc(k)}</dt><dd>${esc(x)}</dd>`).join('');
  let foot = '';
  if (e.review) foot = `<button class="btn primary" data-review-open="${esc(e.review)}">Переглянути сеанс</button>`;
  else if (e.kind === 'pad') foot = '<span class="muted">кадри сеансу вже видалено з пам\'яті сервера (зберігаються 5 останніх)</span>';
  else foot = '<span class="muted">відео моніторингу не записується, зберігається лише знімок у пам\'яті сервера</span>';
  $('#inc-foot').innerHTML = foot;
  $('#incident').hidden = false;
  $('#inc-close').focus();
}

function closeIncident() { $('#incident').hidden = true; }

// Журнал у вкладці «Моніторинг»: рядки перебудовуються лише для нових або змінених подій,
// щоб знімки не завантажувалися щоразу заново.
const incNodes = new Map();
function incidentRow(e) {
  const v = incidentView(e);
  const li = document.createElement('li');
  li.className = `inc ${e.level}`;
  li.dataset.inc = e.id;
  li.tabIndex = 0;
  // у вузькій колонці – коротко: хто, час і число; результат перевірки – на позначці праворуч
  const value = e.kind === 'pad' ? (e.score != null ? `оцінка ${fmt(e.score)}` : FAIL_REASON[e.reason] || '')
    : `подібність ${fmt(e.similarity)}`;
  const title = e.kind === 'unknown' ? v.title : e.kind === 'known' ? e.person_id : `${e.person_id} – перевірка`;
  const badge = e.kind === 'pad' ? { attack: 'атака', failed: 'не пройдено', bona_fide: 'пройдено' }[e.decision] || e.decision
    : LEVEL_TEXT[e.level];
  li.title = `${v.title} · ${v.sub}`;
  li.innerHTML = `${thumb(e)}<div class="txt"><b>${esc(title)}</b><small>${esc(e.time)} · ${esc(value)}</small></div>
    <span class="lvl">${esc(badge)}</span>`;
  return li;
}

function renderIncidents() {
  const ul = $('#m-incidents');
  const only = $('#m-only-alarms').checked;
  const list = App.incidents.filter(e => !only || e.level !== 'info').slice(0, 40);
  if (!list.length) {
    ul.innerHTML = `<li class="empty">${only ? 'тривог ще немає' : 'подій ще немає'}</li>`;
    return;
  }
  const want = list.map(e => {
    const sig = `${e.id}|${e.review || ''}|${e.snapshot}`;
    let li = incNodes.get(e.id);
    if (!li || li.dataset.sig !== sig) {
      li = incidentRow(e);
      li.dataset.sig = sig;
      incNodes.set(e.id, li);
    }
    return li;
  });
  const kids = ul.children;
  if (want.length !== kids.length || want.some((li, i) => kids[i] !== li)) ul.replaceChildren(...want);
  for (const id of incNodes.keys()) if (!App.incidents.some(e => e.id === id)) incNodes.delete(id);
}

// ----- службові -----

async function api(path, opts = {}) {
  const headers = opts.json ? { 'Content-Type': 'application/json' } : {};
  const r = await fetch(path, { method: opts.method || (opts.json || opts.body ? 'POST' : 'GET'), headers,
    body: opts.json ? JSON.stringify(opts.json) : opts.body });
  let data = null;
  try { data = await r.json(); } catch (_) { /* тіло без JSON */ }
  if (!r.ok) {
    const d = data && data.detail;
    // помилки перевірки FastAPI (422) приходять англійською: показується лише назва поля
    const field = x => (x.loc && x.loc.length ? `некоректне значення поля «${x.loc[x.loc.length - 1]}»` : 'некоректні дані');
    const err = new Error(typeof d === 'string' ? d : Array.isArray(d) ? d.map(field).join('; ') : `помилка сервера (код ${r.status})`);
    err.status = r.status;
    // строк токена минув, сесію завершено або обліковий запис заблоковано – знову форма входу
    if (r.status === 401 && !path.startsWith('/api/auth/') && App.user) Auth.showLogin(err.message);
    throw err;
  }
  return data;
}

// ----- вхід до системи (рисунок 4.1) -----

const ROLE_TEXT = { admin: 'адміністратор', operator: 'оператор' };
const isAdmin = () => !Auth.enabled || (App.user && App.user.role === 'admin');

const Auth = {
  enabled: true,

  apply(user) {
    App.user = user;
    document.body.classList.remove('loading');
    const out = this.enabled && !user;
    document.body.classList.toggle('signed-out', out);
    document.body.classList.toggle('role-operator', !isAdmin());
    $('#account').hidden = !user;
    if (user) {
      $('#me-email').textContent = user.email;
      $('#me-role').textContent = user.role_text || ROLE_TEXT[user.role];
    }
  },
  showLogin(message = '') {
    if (Cam.active) { Cam.stop(); cameraChanged(); }
    this.apply(null);
    const box = $('#l-error');
    box.textContent = upperFirst(message);
    box.hidden = !message;
    $('#l-password').value = '';
    ($('#l-email').value ? $('#l-password') : $('#l-email')).focus();
  },
  async submit(ev) {
    ev.preventDefault();
    const email = $('#l-email').value.trim(), password = $('#l-password').value;
    const box = $('#l-error');
    if (!email || !password) {
      box.textContent = 'Вкажіть електронну пошту і пароль';
      box.hidden = false;
      return;
    }
    $('#l-submit').disabled = true;
    try {
      const r = await api('/api/auth/login', { json: { email, password, remember: $('#l-remember').checked } });
      box.hidden = true;
      $('#l-password').value = '';
      this.apply(r.user);
      if (!App.started) await start();
      else { route(); if (!Stream.open) Stream.connect(); }
    } catch (e) {
      box.textContent = upperFirst(e.message);
      box.hidden = false;
      $('#l-password').select();
    } finally {
      $('#l-submit').disabled = false;
    }
  },
  async logout() {
    try { await api('/api/auth/logout', { method: 'POST' }); } catch (_) { /* cookie однаково видаляється з вкладки */ }
    // нова сторінка без стану попереднього користувача: журнал, сеанси, кадри
    location.hash = '';
    location.reload();
  },
};

function toast(msg, kind = '') {
  const el = $('#toast');
  el.textContent = msg;
  el.className = `toast show ${kind}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.className = 'toast'; }, 4500);
}

// ----- камера -----

// причини помилки getUserMedia без англійських повідомлень браузера
const CAM_ERRORS = {
  NotAllowedError: 'доступ до камери заборонено в налаштуваннях браузера',
  NotFoundError: 'камеру не знайдено',
  NotReadableError: 'камеру використовує інша програма',
  OverconstrainedError: 'вибрана камера не підтримує потрібну роздільність',
  SecurityError: 'браузер дозволяє камеру лише для localhost або HTTPS',
};

const Cam = {
  video: null, grab: document.createElement('canvas'), stream: null, active: false, label: '', deviceId: '',

  init() {
    this.video = document.createElement('video');
    this.video.muted = true;
    this.video.playsInline = true;
    this.video.hidden = true;
    document.body.appendChild(this.video);
  },
  async devices() {
    try { return (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput'); } catch (_) { return []; }
  },
  async start(deviceId = '') {
    this.stop();
    const video = { width: { ideal: 1280 }, height: { ideal: 720 } };
    if (deviceId) video.deviceId = { exact: deviceId };
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    } catch (e) {
      toast(`Камера недоступна: ${CAM_ERRORS[e.name] || e.message}`, 'err');
      cameraChanged();
      return;
    }
    this.video.srcObject = this.stream;
    await this.video.play();
    const track = this.stream.getVideoTracks()[0];
    this.label = track.label || 'Камера';
    this.deviceId = track.getSettings().deviceId || deviceId;
    this.active = true;
    await fillDevices();
    cameraChanged();
  },
  async startFile(file) {
    this.stop();
    this.video.srcObject = null;
    this.video.src = URL.createObjectURL(file);
    this.video.loop = true;
    try {
      await this.video.play();
    } catch (e) {
      toast('Файл не вдалося відтворити: браузер не підтримує цей формат відео', 'err');
      this.stop();
      cameraChanged();
      return;
    }
    this.label = `Файл: ${file.name}`;
    this.deviceId = 'file';
    this.active = true;
    cameraChanged();
  },
  stop() {
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    this.stream = null;
    if (this.video.src) { URL.revokeObjectURL(this.video.src); this.video.removeAttribute('src'); this.video.load(); }
    this.active = false;
  },
};

async function fillDevices() {
  const devs = await Cam.devices();
  for (const sel of $$('.source-select')) {
    sel.innerHTML = '';
    if (!Cam.active) sel.add(new Option('Камеру не ввімкнено', ''));
    devs.forEach((d, i) => sel.add(new Option(d.label || `Камера ${i + 1}`, d.deviceId)));
    if (Cam.deviceId === 'file') sel.add(new Option(Cam.label, 'file'));
    sel.add(new Option('Відеофайл…', 'pick-file'));
    sel.value = Cam.deviceId || '';
  }
  $('#cams').innerHTML = devs.filter(d => d.label).map(d => `<option value="${esc(d.label)}">`).join('');
}

function cameraChanged() {
  fillDevices();
  $('#v-cam').textContent = Cam.active ? `● ${Cam.label}` : '● камеру не ввімкнено';
  $('#v-placeholder').hidden = Cam.active;
  $('#m-placeholder').hidden = Cam.active;
  $('#m-cam').textContent = Cam.active ? `● ${Cam.label}` : '● камеру не ввімкнено';
  $('#r-pv-center').hidden = Cam.active;
  $('#r-source-label').textContent = Cam.active ? Cam.label : 'камеру не ввімкнено';
  if (!Cam.active) App.frame = null;
  schedule();
  render();
}

function pickFile() {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'video/*';
  input.onchange = () => input.files[0] && Cam.startFile(input.files[0]);
  input.click();
}

// ----- WebSocket і цикл кадрів -----

const Stream = {
  ws: null, open: false, inflight: false, sentAt: 0, times: [], rtt: null,

  connect() {
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/stream`);
    ws.onopen = () => {
      this.open = true;
      this.inflight = false;
      setLive();
      refreshIncidents(true);   // після перепідключення старі події не сповіщаються ще раз
      if (App.view === 'monitor') this.send({ cmd: 'monitor', on: true });
      schedule();
    };
    ws.onclose = ev => {
      this.open = false;
      this.inflight = false;
      setLive();
      // 4401 – потрібно увійти знову; з'єднання відновлюється після входу
      if (ev.code === 4401) {
        if (App.user) Auth.showLogin('Сесія завершилася: увійдіть знову');
        return;
      }
      // сервер перериває сеанс, якщо з'єднання закрито
      if (isActive(App.session)) refreshSession();
      setTimeout(() => { if (App.user || !Auth.enabled) this.connect(); }, 1500);
    };
    ws.onmessage = ev => onMessage(JSON.parse(ev.data));
    this.ws = ws;
  },
  send(obj) {
    if (this.open) this.ws.send(JSON.stringify(obj));
  },
  tick() {
    const now = performance.now();
    this.rtt = now - this.sentAt;
    this.times.push(now);
    while (this.times.length > 2 && now - this.times[0] > 2000) this.times.shift();
  },
  fps() {
    const n = this.times.length;
    return n > 1 ? (n - 1) * 1000 / (this.times[n - 1] - this.times[0]) : null;
  },
};

function setLive() {
  const el = $('#live');
  el.textContent = Stream.open ? '● LIVE' : '● OFFLINE';
  el.classList.toggle('on', Stream.open);
}

let scheduled = false;
function schedule() {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(sendFrame);
}

async function sendFrame() {
  scheduled = false;
  // під час перегляду запису кадри камери не надсилаються: сервер не витрачає час на обробку
  if (!Cam.active || !Stream.open || Stream.inflight || App.view === 'review') return;
  const v = Cam.video;
  if (v.readyState < 2 || !v.videoWidth) { schedule(); return; }
  const scale = Math.min(1, 1280 / v.videoWidth);
  const w = Math.round(v.videoWidth * scale), h = Math.round(v.videoHeight * scale);
  const g = Cam.grab;
  if (g.width !== w || g.height !== h) { g.width = w; g.height = h; }
  g.getContext('2d').drawImage(v, 0, 0, w, h);
  Stream.inflight = true;
  const t = performance.now() / 1000;
  const jpeg = await new Promise(r => g.toBlob(r, 'image/jpeg', 0.9));
  const head = new ArrayBuffer(8);
  new DataView(head).setFloat64(0, t, true);
  const buf = await new Blob([head, jpeg]).arrayBuffer();
  if (!Stream.open) { Stream.inflight = false; return; }
  Stream.sentAt = performance.now();
  Stream.ws.send(buf);
}

function onMessage(m) {
  if (m.type === 'frame') {
    Stream.inflight = false;
    Stream.tick();
    App.frame = m;
    if (m.session) setSession(m.session);
    if (m.enroll) onEnroll(m.enroll);
    if (m.monitor) ingestEvents(m.monitor.events);
    render();
    schedule();
  } else if (m.type === 'frame_error') {
    Stream.inflight = false;
    schedule();
  } else if (m.type === 'attached') {
    setSession(m.session);
    render();
  } else if (m.type === 'enroll_started') {
    App.enroll = { person_id: m.person_id, progress: 0 };
    render();
  } else if (m.type === 'monitor_started') {
    App.monitor = m.monitor;
    App.frame = null;
    ingestEvents(m.monitor.events);
    render();
  } else if (m.type === 'error') {
    toast(m.message, 'err');
  }
}

// ----- сеанси -----

function setSession(s) {
  const prev = App.session;
  App.session = s;
  if (prev && prev.id === s.id && !FINAL.has(prev.state) && FINAL.has(s.state)) sessionFinished(s);
}

function sessionFinished(s) {
  if (s.mode === 'record') {
    if (s.state === 'cancelled') toast('Запис перервано: зашифровано лише відео, анотацію не створено', 'err');
    else toast(`Запис збережено (AES-256-GCM): ${s.record_name}`);
    refreshRecordings();
  } else if (s.decision) {
    // результат перевірки потрапляє в журнал інцидентів, звідти ж – сповіщення зі звуком
    refreshIncidents();
  }
}

async function refreshSession() {
  if (!App.session) return;
  try { setSession(await api(`/api/sessions/${App.session.id}`)); } catch (_) { /* сеанс уже видалено */ }
  render();
}

async function startSession(body) {
  if (!Cam.active) { toast('Спершу увімкніть камеру', 'err'); return; }
  if (!Stream.open) { toast('Немає з\'єднання з сервером', 'err'); return; }
  try {
    const s = await api('/api/sessions', { json: body });
    App.session = s;
    App.enroll = null;
    Stream.send({ cmd: 'attach', session_id: s.id });
    render();
  } catch (e) {
    toast(e.message, 'err');
  }
}

async function cancelSession() {
  if (!isActive(App.session)) return;
  try { setSession(await api(`/api/sessions/${App.session.id}/cancel`, { method: 'POST' })); } catch (e) { toast(e.message, 'err'); }
  render();
}

function startVerify() {
  return startSession({ mode: 'verify', claimed_person_id: $('#v-person').value.trim() });
}

// З моніторингу: перевірка розпізнаної особи активними викликами (захист від підміни обличчя).
async function checkPerson(pid) {
  if (isActive(App.session)) { toast('Спершу завершіть поточний сеанс', 'err'); return; }
  $('#v-person').value = pid;
  refreshParticipants();
  location.hash = '#/verify';
  await startSession({ mode: 'verify', claimed_person_id: pid, origin: 'monitor' });
}

// ----- еталони -----

async function refreshParticipants() {
  try { App.participants = await api('/api/participants'); } catch (_) { App.participants = []; }
  $('#persons').innerHTML = App.participants.map(p => `<option value="${esc(p.id)}">`).join('');
  renderPersonInfo();
}

function renderPersonInfo() {
  const id = $('#v-person').value.trim();
  const p = App.participants.find(x => x.id === id);
  let text = p ? `еталон: ${STORED_TEXT[p.stored] || p.stored}` : 'еталон не зареєстровано: s_enr = s_ref';
  if (p && p.stored === 'encrypted' && !App.config.encryption.key_set) text = `еталон: ${STORED_TEXT.locked}`;
  $('#v-person-info').textContent = text;
}

function onEnroll(e) {
  App.enroll = e.done ? null : e;
  if (!e.done) return;
  if (e.ok) {
    toast(`Еталон ${e.person_id} зареєстровано: ${STORED_TEXT[e.stored]}`);
    refreshParticipants();
    if (App.view === 'persons') refreshPersons();
  } else {
    toast(e.message, 'err');
  }
}

function startEnroll() {
  if (!Cam.active) { toast('Спершу увімкніть камеру', 'err'); return; }
  Stream.send({ cmd: 'enroll', person_id: $('#v-person').value.trim() });
}

async function enrollPhoto(file) {
  const id = $('#v-person').value.trim();
  try {
    const r = await api(`/api/participants/${encodeURIComponent(id)}/image`, { body: file });
    toast(`Еталон ${r.person_id} з фото: ${STORED_TEXT[r.stored]}`);
    refreshParticipants();
  } catch (e) {
    toast(e.message, 'err');
  }
}

// ----- малювання кадру -----

function drawFrame(canvas, m, label) {
  const g = Cam.grab;
  const W = g.width, H = g.height;
  if (!W || !Cam.active) {
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext('2d');
  // кадр віддзеркалено, як у дзеркалі; точки приходять у координатах невіддзеркаленого кадру
  ctx.setTransform(-1, 0, 0, 1, W, 0);
  ctx.drawImage(g, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  if (!m) return;
  const k = W / 1280;
  const X = x => W - x;
  if (m.face) {
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    const r = Math.max(1.2, 1.8 * k);
    for (let i = 0; i < m.face.length; i += 4) {
      ctx.beginPath();
      ctx.arc(X(m.face[i]), m.face[i + 1], r, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
  if (m.box) {
    const [x, y, w, h] = m.box;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = Math.max(1.5, 2.5 * k);
    ctx.strokeRect(X(x + w), y, w, h);
    if (label) {
      const fs = Math.round(Math.max(14, 30 * k));
      ctx.font = `600 ${fs}px "Segoe UI", Arial, sans-serif`;
      const tw = ctx.measureText(label).width, pad = 6 * k;
      const lx = X(x + w), ly = Math.max(fs + 2 * pad, y - 4 * k);
      ctx.fillStyle = 'rgba(0,0,0,0.65)';
      ctx.fillRect(lx - 1, ly - fs - 2 * pad, tw + 2 * pad, fs + 2 * pad);
      ctx.fillStyle = '#fff';
      ctx.textBaseline = 'bottom';
      ctx.fillText(label, lx + pad, ly - pad);
    }
  }
  // усі знайдені кисті (до двох); виклик перевіряється за кистю, найближчою до обличчя
  const hands = m.hands && m.hands.length ? m.hands : (m.hand ? [m.hand] : []);
  for (const hand of hands) drawHand(ctx, hand, X, k);
}

const HAND_BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

function drawHand(ctx, hand, X, k) {
  const p = i => [X(hand[2 * i]), hand[2 * i + 1]];
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = Math.max(2, 3 * k);
  ctx.beginPath();
  for (const [a, b] of HAND_BONES) { ctx.moveTo(...p(a)); ctx.lineTo(...p(b)); }
  ctx.stroke();
  ctx.fillStyle = '#1c1c1c';
  for (let i = 0; i < 21; i++) {
    ctx.beginPath();
    ctx.arc(...p(i), Math.max(3, 6 * k), 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  }
}

// Підпис над рамкою обличчя: заявлена особа і поточна подібність s_enr.
function boxLabel(s) {
  const f = App.frame;
  if (!s || !isActive(s)) return f && f.box ? `обличчя · ${fmt(f.box[4])}` : '';
  const ser = s.series, n = ser.s_enr.length;
  let cur = null, prev = null;
  for (let i = n - 1; i >= 0; i--) {
    if (ser.s_enr[i] == null) continue;
    if (cur == null) cur = ser.s_enr[i];
    if (ser.t[n - 1] - ser.t[i] >= 0.5) { prev = ser.s_enr[i]; break; }
  }
  const arrow = cur != null && prev != null ? (cur < prev - 0.05 ? ' ↓' : cur > prev + 0.05 ? ' ↑' : '') : '';
  return `${s.claimed_person_id}? ${fmt(cur)}${arrow}`;
}

// Рамки моніторингу: зелена з псевдонімом і подібністю або червона «Невідома особа».
function drawMonitor(canvas, m) {
  drawFrame(canvas, null, '');
  const faces = m && m.monitor ? m.monitor.faces : [];
  if (!Cam.active || !faces.length) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, k = W / 1280;
  const X = x => W - x;
  const fs = Math.round(Math.max(14, 28 * k)), pad = 6 * k;
  ctx.font = `600 ${fs}px "Segoe UI", Arial, sans-serif`;
  ctx.textBaseline = 'bottom';
  for (const f of faces) {
    const [x, y, w, h] = f.box;
    const color = f.known ? KNOWN_COLOR : UNKNOWN_COLOR;
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(2, 4 * k);
    ctx.strokeRect(X(x + w), y, w, h);
    const sim = f.similarity != null ? ` · ${fmt(f.similarity)}` : '';
    const label = `${f.known ? f.person_id : 'Невідома особа'}${sim}`;
    const tw = ctx.measureText(label).width;
    const lx = X(x + w), ly = Math.max(fs + 2 * pad, y - 2 * k);
    ctx.fillStyle = color;
    ctx.fillRect(lx - ctx.lineWidth / 2, ly - fs - 2 * pad, tw + 2 * pad, fs + 2 * pad);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, lx + pad, ly - pad);
  }
}

// ----- графік s_ref(t), s_enr(t) -----

function drawChart(canvas, s) {
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (!cw || !ch) return;
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  const L = 34, R = cw - 22, T = 8, B = ch - 22;
  const ser = s ? s.series : { t: [], s_ref: [], s_enr: [] };
  const tLast = ser.t.length ? ser.t[ser.t.length - 1] : 0;
  const tMax = Math.max(tLast, 10), tMin = Math.max(0, tMax - 20);
  let yMin = 0.2;
  for (const arr of [ser.s_ref, ser.s_enr]) for (const v of arr) if (v != null && v < yMin) yMin = v;
  yMin = Math.max(-0.2, Math.floor((yMin - 0.05) * 10) / 10);
  const x = t => L + (t - tMin) / (tMax - tMin) * (R - L);
  const yMax = 1.05;  // запас зверху, щоб лінія біля 1,0 не зливалася з краєм
  const y = v => B - (v - yMin) / (yMax - yMin) * (B - T);
  ctx.font = '12px "Segoe UI", Arial, sans-serif';
  ctx.fillStyle = '#999';
  ctx.textBaseline = 'middle';
  ctx.strokeStyle = '#eee';
  ctx.lineWidth = 1;
  for (let v = Math.ceil(yMin * 5) / 5; v <= 1.0001; v += 0.2) {
    ctx.beginPath(); ctx.moveTo(L, y(v)); ctx.lineTo(R, y(v)); ctx.stroke();
    ctx.fillText(v.toFixed(1), 4, y(v));
  }
  ctx.textBaseline = 'top';
  for (let t = Math.ceil(tMin / 2) * 2; t <= tMax; t += 2) ctx.fillText(`${t} с`, x(t) - 8, B + 5);
  if (!s) {
    ctx.fillStyle = '#888';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = '13.5px "Segoe UI", Arial, sans-serif';
    ctx.fillText('графік з\'явиться після початку сеансу', (L + R) / 2, (T + B) / 2);
    ctx.textAlign = 'left';
    return;
  }
  // вікна викликів: «під час» – сіре тло, «до» і «після» – світліше
  const st = s.settings;
  for (const [a, b] of s.spans) {
    ctx.fillStyle = '#f6f6f6';
    ctx.fillRect(x(Math.max(tMin, a - st.before_s)), T, Math.max(0, x(a) - x(Math.max(tMin, a - st.before_s))), B - T);
    ctx.fillRect(x(b), T, Math.max(0, Math.min(x(b + st.after_s), R) - x(b)), B - T);
    ctx.fillStyle = '#e9e9e9';
    ctx.fillRect(x(Math.max(tMin, a)), T, Math.max(0, x(b) - x(Math.max(tMin, a))), B - T);
  }
  const last = s.spans[s.spans.length - 1];
  if (last && x(last[0]) > L + 40) {
    ctx.fillStyle = '#777';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('до', x(last[0] - st.before_s / 2), B - 3);
    ctx.fillText('під час', (x(last[0]) + x(last[1])) / 2, B - 3);
    if (x(last[1] + st.after_s / 2) < R) ctx.fillText('після', x(last[1] + st.after_s / 2), B - 3);
    ctx.textAlign = 'left';
  }
  const line = (arr, color, dash, width) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] == null || ser.t[i] < tMin) { pen = false; continue; }
      const px = x(ser.t[i]), py = y(Math.min(1, arr[i]));
      if (pen) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      pen = true;
    }
    ctx.stroke();
    ctx.setLineDash([]);
  };
  line(ser.s_enr, '#8a8a8a', [5, 4], 1.6);
  line(ser.s_ref, '#000', [], 2);
}

// ----- форма 4.7 -----

function instructionView(s) {
  if (App.enroll) {
    return { title: `Реєстрація еталона ${App.enroll.person_id}`, sub: 'Дивіться прямо в камеру, не рухайтеся', progress: App.enroll.progress };
  }
  if (!s) {
    return Cam.active
      ? { title: 'Сеанс не розпочато', sub: 'Вкажіть заявлену особу і натисніть «Почати сеанс»', progress: 0 }
      : { title: 'Камеру не ввімкнено', sub: 'Увімкніть вебкамеру або виберіть відеофайл', progress: 0 };
  }
  const st = s.status;
  if (s.state === 'challenge') {
    const cur = s.challenges[st.index];
    const kind = cur.type.startsWith('cover_') ? 'cover' : cur.type;
    return { title: `Виклик ${st.index + 1} з ${st.count}: ${lowerFirst(st.instruction)}`, sub: SUBTEXT[kind],
      progress: st.progress, remaining: st.remaining, duration: s.challenge_duration_s };
  }
  if (s.state === 'features') {
    return { title: st.instruction, sub: `Збір вікна «після» (${fmt(s.settings.after_s, 1)} с) і обчислення ознак`, progress: 1 };
  }
  if (s.state === 'reference' || s.state === 'created') {
    return { title: st.instruction, sub: 'Фіксація еталона сеансу за кадрами з упевненою детекцією', progress: 0 };
  }
  if (s.state === 'generating') return { title: st.instruction, sub: 'Генерування випадкових викликів', progress: 0 };
  if (s.state === 'cancelled') return { title: 'Сеанс перервано', sub: 'Натисніть «Повторити», щоб почати новий сеанс', progress: 0 };
  return { title: st.instruction, sub: 'Сеанс завершено. «Повторити» – новий сеанс з новими викликами', progress: 1 };
}

function measureText(s) {
  const m = s && s.measure;
  if (!m || s.state !== 'challenge') return null;
  const st = s.settings;
  if (m.type.startsWith('cover_')) {
    const sideOk = m.type === 'cover_mouth' ? m.lower : m.side === m.type.slice(6);
    const covOk = m.coverage >= st.min_coverage;
    const side = m.type === 'cover_mouth' ? `Нижня частина: ${m.lower ? 'так ✓' : 'ні'}` : `Сторона: ${SIDE_TEXT[m.side] || '–'}${sideOk ? ' ✓' : ''}`;
    return `Перекриття обличчя кистю: <b>${Math.round(m.coverage * 100)} %</b> (поріг ${Math.round(st.min_coverage * 100)} %)${covOk ? ' ✓' : ''} · ${side}`;
  }
  if (m.type === 'turn_head') {
    return `Поворот голови: yaw <b>${m.yaw == null ? '–' : Math.round(m.yaw) + '°'}</b> (поріг ${Math.round(st.min_yaw_deg)}°)${m.ok ? ' ✓' : ''}`;
  }
  if (m.offset == null) return 'Кисть не виявлено · проведіть рукою через середню лінію обличчя';
  return `Кисть ${m.offset > 0 ? 'ліворуч' : 'праворуч'} від середньої лінії: <b>${fmt(Math.abs(m.offset))}</b> ширини обличчя`;
}

function renderVerify() {
  const s = App.session && App.session.mode === 'verify' ? App.session : null;
  const f = App.frame;
  drawFrame($('#v-canvas'), f, boxLabel(s));

  // відео: інструкція, таймер, індикатор, вимірювання
  const iv = instructionView(s);
  $('#v-instr-title').textContent = iv.title;
  $('#v-instr-sub').textContent = iv.sub || '';
  $('#v-progress').style.width = `${Math.round((iv.progress || 0) * 100)}%`;
  const timer = $('#v-timer');
  timer.hidden = iv.remaining == null;
  if (iv.remaining != null) {
    $('#v-remaining').textContent = `${fmt(iv.remaining, 1)}с`;
    const c = 2 * Math.PI * 34;
    $('#v-ring').style.strokeDashoffset = String(c * (1 - iv.remaining / (iv.duration || 5)));
  }
  const chip = measureText(s);
  $('#v-chip').hidden = !chip;
  if (chip) $('#v-chip').innerHTML = chip;
  const fps = Stream.fps();
  $('#v-fps').textContent = f ? `FPS ${fmt(fps, 1)} · ${Math.round(f.timings.total_ms + (f.pad_ms ? f.pad_ms - f.timings.landmarks_ms : 0))} мс` : 'FPS – · – мс';

  // ліва колонка
  $('#v-code').textContent = s ? s.code : '–';
  $('#v-back').hidden = !(s && s.origin === 'monitor');
  $('#v-state').textContent = s ? `стан: ${s.state_text}` : (App.enroll ? 'реєстрація еталона…' : 'сеанс не розпочато');
  const list = $('#v-challenges');
  if (s && s.challenges.length) {
    list.innerHTML = s.challenges.map((c, i) => {
      const ico = c.state === 'done' ? '✓' : c.state === 'failed' ? '×' : i + 1;
      const sc = c.score != null ? `<span class="sc">${fmt(c.score)}</span>` : '';
      return `<li class="${c.state}"><span class="ico">${ico}</span><span>${esc(c.instruction)}</span>${sc}</li>`;
    }).join('');
  } else {
    list.innerHTML = `<li class="empty">${s && isActive(s) ? 'фіксація еталона…' : 'з\'являться після фіксації еталона'}</li>`;
  }
  const att = $('#v-attempt');
  att.hidden = !(s && s.status.attempt > 1);
  if (s) att.textContent = `спроба ${s.status.attempt} з ${s.status.max_attempts}: попередній набір не виконано`;
  const c = f ? f.check : null;
  $('#c-face').textContent = c && c.face_points ? `${c.face_points} т.` : '–';
  $('#c-hand').textContent = c && c.hand_points ? (c.hands > 1 ? `${c.hands} кисті × ${c.hand_points} т.` : `${c.hand_points} т.`) : '–';
  $('#c-pose').textContent = c && c.yaw != null ? `${Math.round(c.yaw)}° / ${Math.round(c.pitch)}°` : '–';
  $('#c-det').textContent = c ? (c.embedding ? fmt(c.det_score) : 'немає') : '–';

  // графік
  drawChart($('#v-chart'), s);

  // рішення
  const d = $('#v-decision');
  const thr = s ? s.settings.threshold : (App.config ? App.config.settings.threshold : null);
  let cls = 'idle', caption = 'Рішення формується після останнього виклику', title = 'ОЧІКУВАННЯ СЕАНСУ', score = null, lat = '';
  if (s && s.decision) {
    cls = s.decision.decision;
    caption = `Рішення сеансу ${s.code}`;
    title = DECISION_TEXT[cls];
    score = s.decision.score;
    const l = s.decision.explanation.latency || {};
    if (cls === 'failed') lat = s.status.instruction;
    else lat = `обчислення ${fmt(l.decision_compute_ms, 2)} мс · через ${fmt(l.after_last_challenge_s, 2)} с після виклику`;
  } else if (s && s.state === 'cancelled') {
    cls = 'cancelled'; caption = `Сеанс ${s.code}`; title = DECISION_TEXT.cancelled;
  } else if (s && s.preliminary) {
    const p = s.preliminary;
    cls = p.score >= thr ? 'attack' : 'pending';
    caption = `Попередня оцінка (після ${p.n} з ${p.count} викликів)`;
    title = p.score >= thr ? 'ПІДОЗРА НА АТАКУ' : 'ОЗНАК АТАКИ НЕМАЄ';
    score = p.score;
  } else if (s && isActive(s)) {
    cls = 'idle'; caption = 'Сеанс триває'; title = 'ЗБІР ДАНИХ';
  }
  d.className = `decision ${cls}`;
  $('#d-caption').textContent = caption;
  $('#d-title').textContent = title;
  $('#d-score').textContent = fmt(score);
  $('#d-threshold').textContent = fmt(thr);
  $('#d-bar').style.width = `${Math.round((score || 0) * 100)}%`;
  $('#d-lat').textContent = lat;

  // ознаки
  const feats = s && s.features;
  $('#v-features').innerHTML = MAIN_FEATURES.map(([k, name, dig, unit = '']) =>
    `<dt>${name}</dt><dd>${feats ? fmt(feats[k], dig) + unit : '–'}</dd>`).join('');
  $('#v-features-src').textContent = feats ? `виклик: ${lowerFirst(s.features_challenge)}` : 'ознаки обчислюються після кожного виконаного виклику';
  $('#v-allfeat').innerHTML = feats ? Object.entries(feats).map(([k, v]) => `<tr><td>${k}</td><td>${fmt(v, 3)}</td></tr>`).join('') : '';

  // швидкодія
  $('#p-det').textContent = f ? `${fmt(f.timings.detect_ms, 0)} / ${fmt(f.timings.embed_ms, 0)} мс` : '–';
  $('#p-pad').textContent = f && f.pad_ms != null ? `${fmt(f.pad_ms, 1)} мс/кадр` : (f ? `${fmt(f.timings.landmarks_ms, 0)} мс (точки)` : '–');
  const l = s && s.decision && s.decision.explanation.latency;
  $('#p-lat').textContent = l && l.decision_compute_ms != null ? `${fmt(l.decision_compute_ms, 2)} мс` : '–';

  // кнопки
  const active = isActive(s);
  const main = $('#v-main');
  main.textContent = active ? 'Завершити' : 'Почати сеанс';
  main.disabled = !active && (!Cam.active || !Stream.open || !!App.enroll || isActive(App.session));
  $('#v-repeat').disabled = !s || !Cam.active || !!App.enroll;
  $('#v-enroll').disabled = !Cam.active || active || !!App.enroll;
}

// ----- моніторинг -----

let facesKey = null;

function renderMonitor() {
  const f = App.frame && App.frame.monitor ? App.frame : null;
  const mon = f ? f.monitor : null;
  const st = App.monitor;
  drawMonitor($('#m-canvas'), f);
  const fps = Stream.fps();
  $('#m-fps').textContent = f ? `FPS ${fmt(fps, 1)} · ${Math.round(f.timings.total_ms)} мс` : 'FPS – · – мс';

  // база еталонів
  const thr = mon ? mon.threshold : (st ? st.threshold : App.config.monitoring.threshold);
  $('#m-count').textContent = mon ? mon.gallery : (st ? st.ids.length : '–');
  $('#m-ids').innerHTML = st ? st.ids.map(id => `<li>${esc(id)}</li>`).join('') : '';
  const locked = st ? st.locked : [];
  $('#m-locked').textContent = locked.length
    ? `ще ${locked.length} зашифровано, ключ PAD_DATASET_KEY не задано або інший: ${locked.join(', ')}`
    : (st && !st.ids.length ? 'база порожня – зареєструйте еталон на вкладці «Верифікація»' : '');
  $('#m-threshold').textContent = fmt(thr);
  $('#m-backend').textContent = st ? st.backend.split(' ')[0] : App.config.monitoring.backend.split(' ')[0];
  $('#m-backend').title = st ? st.backend : App.config.monitoring.backend;
  $('#m-embedder').textContent = (App.config.embedder || '–').split(' ')[0].replace(/_/g, ' ');

  // сповіщення про невідому особу
  const alert = mon && mon.alert;
  $('#m-alert').hidden = !alert;
  if (alert) {
    const who = alert.count > 1 ? `${alert.count} облич немає` : 'обличчя немає';
    const sim = alert.similarity != null ? ` · подібність ${fmt(alert.similarity)} < ${fmt(thr)}` : '';
    $('#m-alert-sub').textContent = mon.gallery ? `${who} в базі еталонів${sim}` : 'база еталонів порожня';
  }

  // обличчя в кадрі: список перебудовується лише тоді, коли змінився склад облич,
  // інакше кнопка «Перевірити» зникала б між натисканням і відпусканням
  const faces = mon ? mon.faces : [];
  const key = faces.map(x => (x.known ? x.person_id : '?')).join('|');
  const list = $('#m-faces');
  if (key !== facesKey) {
    facesKey = key;
    list.innerHTML = faces.length ? faces.map(x => x.known
      ? `<li class="known"><i></i><span class="who">${esc(x.person_id)}</span><span class="sim"></span>
         <button class="btn sm" data-check="${esc(x.person_id)}" title="Сеанс верифікації з активними викликами для ${esc(x.person_id)}">Перевірити</button></li>`
      : `<li class="unknown"><i></i><span class="who">Невідома особа<small></small></span><span class="sim"></span></li>`).join('')
      : `<li class="empty">${Cam.active ? 'облич не виявлено' : 'камеру не ввімкнено'}</li>`;
  }
  $$('#m-faces li:not(.empty)').forEach((li, i) => {
    const x = faces[i];
    if (!x) return;
    $('.sim', li).textContent = fmt(x.similarity);
    const small = $('small', li);
    if (small) small.textContent = x.nearest ? `найближчий еталон: ${x.nearest}` : '';
  });

  // журнал інцидентів
  renderIncidents();

  // швидкодія
  $('#m-p-det').textContent = f ? `${fmt(f.timings.detect_ms, 0)} / ${fmt(f.timings.embed_ms, 0)} мс` : '–';
  $('#m-p-search').textContent = f && f.timings.search_ms != null ? `${fmt(f.timings.search_ms, 2)} мс` : '–';
}

// ----- форма 4.8 -----

function recordParams() {
  const types = $$('#r-types input:checked').map(i => i.value);
  const ptype = $('input[name=ptype]:checked').value;
  return {
    participant: $('#r-participant').value.trim(),
    consent: $('#r-consent').checked,
    consent_ref: $('#r-consent-ref').value.trim(),
    presentation_type: ptype,
    swap_tool: ptype === 'attack' ? $('#r-tool').value : '',
    target_face: ptype === 'attack' ? $('#r-target').value.trim() : '',
    lighting: $('#r-lighting').value,
    webcam: $('#r-webcam').value.trim(),
    source: Cam.active ? Cam.label : '',
    scenario: { challenge_types: types, challenge_count: +$('#r-count').value, duration_s: +$('#r-duration').value },
  };
}

function recordProblem(p) {
  if (!App.config.encryption.key_set) return 'Ключ PAD_DATASET_KEY не задано: записи зберігаються лише зашифрованими, запис недоступний.';
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$/.test(p.participant)) return 'Вкажіть псевдонім учасника (латиницею, наприклад P01), без імені та прізвища.';
  if (!p.consent) return 'Без позначки про письмову згоду учасника запис заборонено.';
  if (p.presentation_type === 'attack' && !p.swap_tool) return 'Для атаки виберіть інструмент підміни.';
  if (p.scenario.challenge_types.length < 4) return 'Позначте не менше 4 типів викликів (вимога ТЗ).';
  if (!Cam.active) return 'Увімкніть камеру або виберіть джерело відео.';
  if (!Stream.open) return 'Немає з\'єднання з сервером.';
  return null;
}

function renderRecord() {
  const s = App.session && App.session.mode === 'record' ? App.session : null;
  const recording = isActive(s);
  const cv = $('#r-canvas');
  drawFrame(cv, App.frame, '');
  cv.classList.toggle('blur', $('#r-blur').checked);

  const p = recordParams();
  const problem = recordProblem(p);
  const btn = $('#r-start');
  btn.textContent = recording ? '■ Зупинити запис' : '● Почати запис';
  btn.classList.toggle('rec', recording);
  btn.disabled = !recording && (!!problem || isActive(App.session));
  const hint = $('#r-hint');
  const claimed = p.presentation_type === 'attack' && p.target_face ? p.target_face : p.participant;
  hint.className = `hint ${problem && !recording ? 'warn' : ''}`;
  hint.textContent = recording ? `Іде запис сеансу ${s.code} → ${s.record_name}`
    : problem || `Заявлена особа: ${claimed || '–'}. Відео і анотація будуть зашифровані після завершення сеансу.`;
  const attack = p.presentation_type === 'attack';
  $('#r-tool').disabled = !attack || recording;
  $('#r-target').disabled = !attack || recording;
  $$('#r-participant, #r-webcam, #r-lighting, #r-consent, #r-consent-ref, input[name=ptype], #r-count, #r-duration, #r-types input')
    .forEach(el => { el.disabled = recording; });

  const label = $('#r-pv-label');
  label.classList.toggle('rec', recording);
  label.firstChild.textContent = recording ? '● REC · ' : '○ Попередній перегляд · ';
  const box = $('#r-pv-instr');
  box.hidden = !s;
  if (s) {
    const iv = instructionView(s);
    $('#r-instr').textContent = iv.title;
    $('#r-remaining').textContent = iv.remaining != null ? `${fmt(iv.remaining, 1)} с` : (recording ? s.state_text : '');
    $('#r-progress').style.width = `${Math.round((iv.progress || 0) * 100)}%`;
  }
}

function conditionText(r) {
  const res = r.resolution ? `${r.resolution.split('x')[1]}p` : '–';
  return `${LIGHT_SHORT[r.lighting] || r.lighting || '–'} · ${res}`;
}

function statusCell(r) {
  if (r.status === 'reviewed') return `<button class="status" data-review="${esc(r.name)}" data-to="0" title="Зняти позначку">Розмічено</button>`;
  if (r.status === 'pending') return `<button class="status pending" data-review="${esc(r.name)}" data-to="1" title="Позначити як перевірений і розмічений">Очікує перевірки</button>`;
  const text = { failed: 'Не пройдено', incomplete: 'Перервано', locked: 'Інший ключ' }[r.status] || r.status;
  return `<span class="status static">${text}</span>`;
}

async function refreshRecordings() {
  const body = $('#r-table tbody');
  try {
    App.recordings = await api('/api/recordings');
  } catch (e) {
    App.recordings = null;
    body.innerHTML = `<tr class="empty"><td colspan="10">${esc(e.message)}</td></tr>`;
    return;
  }
  const { rows, summary } = App.recordings;
  const tools = App.config.swap_tools;
  // перерваний сеанс (лише відео без анотації) переглянути не можна – кнопки для нього немає
  const viewCell = r => (r.reviewable ? `<td><button class="btn sm" data-open-review="${esc(r.name)}"
    title="Відео сеансу з графіком подібності і викликами">Переглянути</button></td>` : '<td></td>');
  body.innerHTML = rows.length ? rows.map(r => {
    if (!r.presentation_type) {
      return `<tr><td class="mono">${r.id}</td><td>${esc(r.participant)}</td><td colspan="6" class="muted">${esc(r.name)}</td><td>${statusCell(r)}</td>${viewCell(r)}</tr>`;
    }
    const attack = r.presentation_type === 'attack';
    const sess = r.layout === 'imported' && r.session_index ? ` <span class="muted">· с${r.session_index}</span>` : '';
    return `<tr title="${esc(r.name)}${r.score != null ? ' · оцінка ' + fmt(r.score) : ''}">
      <td class="mono">${r.id}</td><td>${esc(r.participant)}${sess}</td>
      <td><span class="badge ${attack ? 'attack' : ''}">${attack ? 'Атака' : 'Справжнє'}</span></td>
      <td>${attack ? esc(tools[r.swap_tool] || r.swap_tool) : '–'}</td>
      <td>${esc(r.target_face) || '–'}</td>
      <td>${r.challenges.map(esc).join(' · ') || '–'}</td>
      <td>${conditionText(r)}</td>
      <td class="mono">${r.duration_s != null ? fmt(r.duration_s, 1) + ' с' : '–'}</td>
      <td>${statusCell(r)}</td>${viewCell(r)}</tr>`;
  }).join('') : '<tr class="empty"><td colspan="10">Записів ще немає</td></tr>';
  $('#s-participants').textContent = summary.participants;
  $('#s-sessions').textContent = summary.sessions;
  $('#s-bona').textContent = summary.bona_fide;
  $('#s-attack').textContent = summary.attack;
  $('#s-tools').textContent = summary.tools;
  $('#s-targets').textContent = summary.targets;
  $('#s-reviewed').textContent = summary.reviewed;
}

async function review(name, reviewed) {
  try {
    await api(`/api/recordings/${encodeURIComponent(name)}/review`, { json: { reviewed } });
    refreshRecordings();
  } catch (e) {
    toast(e.message, 'err');
  }
}

// ----- перегляд сеансу: відео, графік з курсором часу, виклики і рішення -----

const LAYOUT_TEXT = {
  live: 'живий запис (веб-інтерфейс)', imported: 'віддалений запис (імпорт)', memory: 'сеанс верифікації (пам\'ять сервера)',
};

// Кадри сеансу зберігаються в пам'яті сторінки як ArrayBuffer (не в кеші браузера на диску),
// для показу кадр декодується в ImageBitmap; декодованих кадрів тримається не більше BITMAPS.
const Review = {
  uid: null, meta: null, bufs: [], bitmaps: new Map(), loaded: 0, playing: false, speed: 1,
  tau: 0, end: 0, lastTs: 0, shown: -1, token: 0, chartMap: null, BITMAPS: 40,

  async open(uid) {
    this.close();
    const token = ++this.token;
    this.uid = uid;
    this.message('Розшифрування запису в пам\'яті сервера і підготовка кадрів…');
    $('#rv-uid').textContent = uid;
    $('#rv-title').textContent = 'Запис сеансу';
    let meta;
    try {
      meta = await api(`/api/review/${encodeURIComponent(uid)}`);
    } catch (e) {
      if (token !== this.token) return;
      this.message(`Сеанс неможливо показати: ${e.message}`);
      this.renderInfo(null);
      return;
    }
    if (token !== this.token) return;
    this.meta = meta;
    const ft = meta.frames.t, st = meta.series.t;
    this.end = Math.max(meta.end_s || 0, ft.length ? ft[ft.length - 1] : 0, st.length ? st[st.length - 1] : 0, 0.1);
    this.tau = 0;
    this.bufs = new Array(meta.frames.count);
    this.renderInfo(meta);
    this.update();
    if (!meta.frames.count) {
      this.message(meta.frames.missing ? 'Відео запису не знайдено – показано лише графік і виклики'
        : 'У вікні сеансу немає кадрів відео');
      $('#rv-scrub').disabled = false;
      return;
    }
    this.message(`Завантаження кадрів: 0 з ${meta.frames.count}`);
    let next = 0;
    const worker = async () => {
      while (next < meta.frames.count && token === this.token) {
        const i = next++;
        try {
          const r = await fetch(`/api/review/${encodeURIComponent(uid)}/frame/${i}.jpg`, { cache: 'no-store' });
          if (!r.ok) continue;
          const buf = await r.arrayBuffer();
          if (token !== this.token) return;
          this.bufs[i] = buf;
          this.loaded++;
        } catch (_) { continue; }
        if (this.loaded === 1 || i === 0) { $('#rv-placeholder').hidden = true; this.draw(true); }
        if (this.loaded % 10 === 0) $('#rv-title').textContent = `Завантаження кадрів: ${this.loaded} з ${meta.frames.count}`;
      }
    };
    await Promise.all(Array.from({ length: 6 }, worker));
    if (token !== this.token) return;
    $('#rv-title').textContent = this.title();
    $('#rv-play').disabled = false;
    $('#rv-scrub').disabled = false;
    this.draw(true);
  },

  close() {
    this.token++;
    this.pause();
    for (const b of this.bitmaps.values()) b.close && b.close();
    this.bitmaps.clear();
    Object.assign(this, { uid: null, meta: null, bufs: [], loaded: 0, tau: 0, shown: -1 });
    $('#rv-play').disabled = true;
    $('#rv-scrub').disabled = true;
    $('#rv-chip').hidden = true;
    $('#rv-sim').hidden = true;
    const c = $('#rv-canvas');
    c.getContext('2d').clearRect(0, 0, c.width, c.height);
  },

  message(text) {
    $('#rv-message').textContent = text;
    $('#rv-placeholder').hidden = false;
  },

  title() {
    const m = this.meta;
    if (!m) return 'Запис сеансу';
    const who = m.participant ? `учасник ${m.participant}` : `заявлена особа ${m.claimed_identity || '–'}`;
    return `${LAYOUT_TEXT[m.layout] || m.layout} · ${who} · кадрів ${m.frames.count}`;
  },

  play() {
    if (!this.meta || this.playing) return;
    if (this.tau >= this.end - 1e-3) this.tau = 0;
    this.playing = true;
    this.lastTs = performance.now();
    $('#rv-play').textContent = '❚❚ Пауза';
    requestAnimationFrame(ts => this.tick(ts));
  },
  pause() {
    this.playing = false;
    $('#rv-play').textContent = '▶ Відтворити';
  },
  toggle() { if (this.playing) this.pause(); else this.play(); },
  tick(ts) {
    if (!this.playing) return;
    this.tau = Math.min(this.end, this.tau + Math.max(0, ts - this.lastTs) / 1000 * this.speed);
    this.lastTs = ts;
    if (this.tau >= this.end) this.pause();
    this.update();
    if (this.playing) requestAnimationFrame(t => this.tick(t));
  },
  seek(tau) {
    if (!this.meta) return;
    this.tau = Math.max(0, Math.min(this.end, tau));
    this.update();
  },
  setSpeed(v) {
    this.speed = v;
    $$('.speed [data-speed]').forEach(b => b.classList.toggle('on', +b.dataset.speed === v));
  },

  // Останній кадр з міткою часу не пізніше tau (бінарний пошук).
  frameAt(tau) {
    const t = this.meta ? this.meta.frames.t : [];
    if (!t.length) return -1;
    let lo = 0, hi = t.length - 1;
    if (tau <= t[0]) return 0;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (t[mid] <= tau) lo = mid; else hi = mid - 1;
    }
    return lo;
  },

  async bitmap(i) {
    if (this.bitmaps.has(i)) return this.bitmaps.get(i);
    const buf = this.bufs[i];
    if (!buf) return null;
    const bmp = await createImageBitmap(new Blob([buf], { type: 'image/jpeg' }));
    this.bitmaps.set(i, bmp);
    while (this.bitmaps.size > this.BITMAPS) {
      const [k, old] = this.bitmaps.entries().next().value;
      if (old.close) old.close();
      this.bitmaps.delete(k);
    }
    return bmp;
  },

  async draw(force = false) {
    let i = this.frameAt(this.tau);
    if (i < 0) return;
    // кадр ще не завантажено – показується найближчий попередній
    while (i > 0 && !this.bufs[i]) i--;
    if (!this.bufs[i] || (i === this.shown && !force)) return;
    const token = this.token;
    let bmp;
    try { bmp = await this.bitmap(i); } catch (_) { return; }
    if (!bmp || token !== this.token) return;
    const c = $('#rv-canvas');
    if (c.width !== bmp.width || c.height !== bmp.height) { c.width = bmp.width; c.height = bmp.height; }
    c.getContext('2d').drawImage(bmp, 0, 0);
    this.shown = i;
  },

  // Виклик, у вікні якого зараз курсор, і текст підказки над відео.
  current() {
    const m = this.meta, tau = this.tau;
    const chs = m.challenges.filter(c => c.t_start != null && c.t_end != null);
    const cur = chs.find(c => tau >= c.t_start && tau <= c.t_end);
    if (cur) {
      const same = chs.filter(c => c.attempt === cur.attempt);
      const att = Math.max(...chs.map(c => c.attempt)) > 1 ? `спроба ${cur.attempt}, ` : '';
      return { c: cur, text: `${att}виклик ${same.indexOf(cur) + 1} з ${same.length}: ${cur.name} · ${cur.performed ? 'виконано ✓' : 'не виконано ×'}` };
    }
    if (!chs.length || tau < chs[0].t_start) return { c: null, text: 'Фіксація еталона сеансу: погляд у камеру' };
    if (tau > chs[chs.length - 1].t_end) return { c: null, text: 'Вікно «після» останнього виклику' };
    return { c: null, text: 'Між викликами' };
  },

  update() {
    const m = this.meta;
    if (!m) return;
    $('#rv-clock').textContent = `${fmt(this.tau, 1)} / ${fmt(this.end, 1)} с`;
    const scrub = $('#rv-scrub');
    if (document.activeElement !== scrub || this.playing) scrub.value = String(Math.round(this.tau / this.end * 1000));
    const cur = this.current();
    $('#rv-chip').hidden = false;
    $('#rv-chip').textContent = cur.text;
    // поточна подібність s_ref у момент кадру
    const st = m.series.t;
    let k = -1;
    for (let i = 0; i < st.length && st[i] <= this.tau + 1e-6; i++) k = i;
    const sref = k >= 0 ? m.series.s_ref[k] : null;
    $('#rv-sim').hidden = sref == null;
    if (sref != null) $('#rv-sim').textContent = `s_ref ${fmt(sref)} · s_enr ${fmt(m.series.s_enr[k])}`;
    $$('#rv-challenges li[data-i]').forEach(li => li.classList.toggle('current', cur.c != null && +li.dataset.i === m.challenges.indexOf(cur.c)));
    drawReviewChart($('#rv-chart'), this);
    this.draw();
  },

  renderInfo(m) {
    const info = $('#rv-info');
    if (!m) {
      info.innerHTML = '';
      $('#rv-challenges').innerHTML = '<li class="empty">–</li>';
      $('#rv-truth').textContent = '–';
      setReviewDecision(null);
      drawReviewChart($('#rv-chart'), null);
      return;
    }
    const rows = [];
    if (m.participant) rows.push(['Учасник', m.participant]);
    rows.push(['Заявлена особа', m.claimed_identity || '–']);
    if (m.session_index) rows.push(['Сеанс у записі', `№ ${m.session_index}`]);
    if (m.lighting) rows.push(['Освітлення', LIGHT_SHORT[m.lighting] || m.lighting]);
    rows.push(['Тривалість', m.duration_s != null ? `${fmt(m.duration_s, 1)} с` : '–']);
    rows.push(['Кадрів відео', m.frames.count || 'немає']);
    if (m.frames.width) rows.push(['Розмір кадру', `${m.frames.width}×${m.frames.height}`]);
    rows.push(['Виклики виконано', `${m.performed} з ${m.count}`]);
    info.innerHTML = rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
    $('#rv-challenges').innerHTML = m.challenges.length ? m.challenges.map((c, i) => {
      const sc = c.score != null ? `оцінка ${fmt(c.score)}` : c.model_score != null ? `оцінка моделі ${fmt(c.model_score)}` : '';
      const span = c.t_start != null ? `${fmt(c.t_start, 1)}–${fmt(c.t_end, 1)} с` : '–';
      const att = c.attempt > 1 ? `спроба ${c.attempt} · ` : '';
      return `<li data-i="${i}" class="${c.performed ? 'done' : 'failed'}" title="Перейти до виклику">
        <span class="ico">${c.performed ? '✓' : '×'}</span>
        <span>${esc(c.name)}<small>${att}${span} · ${c.performed ? 'виконано' : 'не виконано'}${sc ? ' · ' + sc : ''}</small></span></li>`;
    }).join('') : '<li class="empty">викликів немає</li>';
    // мітка набору даних: що насправді було в записі
    let truth = 'Живий сеанс верифікації – мітки набору немає';
    if (m.presentation_type === 'attack') {
      const tool = App.config.swap_tools[m.swap_tool] || m.swap_tool || 'інструмент не вказано';
      truth = `<b>Атака підміни обличчя</b><br><span class="muted">${esc(tool)}${m.target_face ? ', цільове обличчя ' + esc(m.target_face) : ''}</span>`;
    } else if (m.presentation_type === 'bona_fide') {
      truth = '<b>Справжнє подання</b>';
    }
    const d = m.decision.decision;
    if (m.presentation_type && (d === 'attack' || d === 'bona_fide')) {
      const ok = (d === 'attack') === (m.presentation_type === 'attack');
      truth += `<div class="match ${ok ? 'ok' : 'bad'}">${ok ? 'Рішення збігається з міткою' : 'Рішення не збігається з міткою'}</div>`;
    }
    $('#rv-truth').innerHTML = truth;
    setReviewDecision(m);
  },
};

function setReviewDecision(m) {
  const d = m ? m.decision : null;
  let cls = 'idle', caption = '–', title = '–', note = '';
  if (d) {
    if (d.source === 'none') {
      caption = 'Запис без оцінки';
      title = 'ОЦІНКИ НЕМАЄ';
      note = d.note;
    } else {
      cls = d.decision === 'offline' ? 'idle' : d.decision;
      caption = d.source === 'model' ? `Оцінка поточною моделлю (${d.model_version || '?'})` : 'Рішення під час сеансу';
      title = DECISION_TEXT[d.decision] || 'ОЦІНКИ НЕМАЄ';
      note = d.source === 'model' ? 'під час запису оцінка не обчислювалася' : d.note;
      if (d.decision === 'offline') note = d.note;
    }
  }
  $('#rv-decision').className = `decision ${cls}`;
  $('#rv-d-caption').textContent = caption;
  $('#rv-d-title').textContent = title;
  $('#rv-d-score').textContent = fmt(d ? d.score : null);
  $('#rv-d-threshold').textContent = fmt(d ? d.threshold : null);
  $('#rv-d-bar').style.width = `${Math.round(((d && d.score) || 0) * 100)}%`;
  $('#rv-d-note').textContent = note || '';
}

// Графік перегляду: увесь сеанс, вікна викликів з назвами, курсор поточного кадру.
function drawReviewChart(canvas, R) {
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (!cw || !ch) return;
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);
  const m = R && R.meta;
  const L = 34, Rt = cw - 14, T = 22, B = ch - 22;
  const tMax = m ? R.end : 10, tMin = 0;
  const ser = m ? m.series : { t: [], s_ref: [], s_enr: [] };
  let yMin = 0.2;
  for (const arr of [ser.s_ref, ser.s_enr]) for (const v of arr) if (v != null && v < yMin) yMin = v;
  yMin = Math.max(-0.2, Math.floor((yMin - 0.05) * 10) / 10);
  const yMax = 1.05;
  const x = t => L + (t - tMin) / (tMax - tMin) * (Rt - L);
  const y = v => B - (v - yMin) / (yMax - yMin) * (B - T);
  if (R) R.chartMap = { L, R: Rt, tMin, tMax };
  ctx.font = '12px "Segoe UI", Arial, sans-serif';
  ctx.fillStyle = '#999';
  ctx.textBaseline = 'middle';
  ctx.strokeStyle = '#eee';
  ctx.lineWidth = 1;
  for (let v = Math.ceil(yMin * 5) / 5; v <= 1.0001; v += 0.2) {
    ctx.beginPath(); ctx.moveTo(L, y(v)); ctx.lineTo(Rt, y(v)); ctx.stroke();
    ctx.fillText(v.toFixed(1), 4, y(v));
  }
  ctx.textBaseline = 'top';
  const step = tMax > 40 ? 5 : tMax > 16 ? 2 : 1;
  for (let t = 0; t <= tMax + 1e-6; t += step) ctx.fillText(`${t} с`, Math.min(x(t) - 8, Rt - 24), B + 5);
  if (!m) return;
  const st = m.settings;
  const chs = m.challenges.filter(c => c.t_start != null && c.t_end != null);
  // вікна «до» і «після» – світле тло, «під час» – сіре; невиконаний виклик – штрихування
  for (const c of chs) {
    const a = x(Math.max(tMin, c.t_start - st.before_s)), b = x(c.t_start), e = x(c.t_end);
    const f = x(Math.min(tMax, c.t_end + st.after_s));
    ctx.fillStyle = '#f6f6f6';
    ctx.fillRect(a, T, Math.max(0, b - a), B - T);
    ctx.fillRect(e, T, Math.max(0, f - e), B - T);
  }
  chs.forEach((c, k) => {
    const a = x(c.t_start), b = x(c.t_end);
    ctx.fillStyle = c.performed ? '#e3e3e3' : '#efefef';
    ctx.fillRect(a, T, Math.max(1, b - a), B - T);
    if (!c.performed) {
      ctx.save();
      ctx.beginPath(); ctx.rect(a, T, Math.max(1, b - a), B - T); ctx.clip();
      ctx.strokeStyle = '#cfcfcf';
      for (let s = a - (B - T); s < b; s += 8) { ctx.beginPath(); ctx.moveTo(s, B); ctx.lineTo(s + (B - T), T); ctx.stroke(); }
      ctx.restore();
    }
    // підпис над вікном: номер, коротка назва, виконано чи ні; якщо не вміщається – лише номер
    ctx.fillStyle = c.performed ? '#111' : '#777';
    ctx.textBaseline = 'top';
    ctx.font = '600 12px "Segoe UI", Arial, sans-serif';
    const mark = c.performed ? '✓' : '×';
    const prevEnd = k > 0 ? x(chs[k - 1].t_end) + 4 : L;
    const nextStart = k + 1 < chs.length ? x(chs[k + 1].t_start) - 4 : Rt;
    const room = nextStart - Math.max(prevEnd, a - 30);
    let label = `${k + 1}. ${c.short} ${mark}`;
    if (ctx.measureText(label).width > room) label = `${k + 1} ${mark}`;
    const lw = ctx.measureText(label).width;
    const lx = Math.max(L, Math.min(Rt - lw, (a + b) / 2 - lw / 2));
    ctx.fillText(label, lx, 4);
  });
  ctx.font = '12px "Segoe UI", Arial, sans-serif';
  const line = (arr, color, dash, width) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < arr.length; i++) {
      if (arr[i] == null) { pen = false; continue; }
      const px = x(ser.t[i]), py = y(Math.min(1, arr[i]));
      if (pen) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      pen = true;
    }
    ctx.stroke();
    ctx.setLineDash([]);
  };
  line(ser.s_enr, '#8a8a8a', [5, 4], 1.6);
  line(ser.s_ref, '#000', [], 2);
  // курсор поточного кадру
  const cx = x(R.tau);
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(cx, T - 2); ctx.lineTo(cx, B); ctx.stroke();
  ctx.fillStyle = '#000';
  ctx.beginPath(); ctx.moveTo(cx - 5, T - 6); ctx.lineTo(cx + 5, T - 6); ctx.lineTo(cx, T); ctx.closePath(); ctx.fill();
}

// ----- підтвердження незворотних дій -----

function confirmDialog(title, text, yes = 'Видалити') {
  return new Promise(resolve => {
    $('#cf-title').textContent = title;
    $('#cf-text').textContent = text;
    $('#cf-yes').textContent = yes;
    const box = $('#confirm');
    const done = ok => {
      box.hidden = true;
      $('#cf-yes').onclick = $('#cf-no').onclick = box.onclick = null;
      resolve(ok);
    };
    $('#cf-yes').onclick = () => done(true);
    $('#cf-no').onclick = () => done(false);
    box.onclick = e => { if (e.target === box) done(false); };
    box.hidden = false;
    $('#cf-no').focus();
  });
}

// ----- дашборд -----

const fmtDateTime = iso => iso ? iso.replace('T', ' ').slice(0, 16) : '–';
const STORED_SHORT = { encrypted: 'зашифровано', memory: 'лише в пам\'яті', plain: 'відкритий .npy', locked: 'зашифровано, ключ не задано' };

async function refreshDashboard() {
  let d;
  try { d = await api('/api/dashboard'); } catch (e) { $('#d-sub').textContent = e.message; return; }
  if (App.view !== 'dashboard') return;
  const c = d.counts;
  const tiles = [
    ['Відомі особи', c.known, 'появ працівників з бази', ''],
    ['Невідомі особи', c.unknown, 'тривог моніторингу', c.unknown ? 'alarm' : ''],
    ['Перевірку пройдено', c.pad_bona_fide, 'справжнє подання', c.pad_bona_fide ? 'ok' : ''],
    ['Атак виявлено', c.pad_attack, 'перевірка викликами', c.pad_attack ? 'alarm' : ''],
    ['Не пройдено', c.pad_failed, 'виклики не виконано', c.pad_failed ? 'warn' : ''],
    ['Осіб у базі', d.persons, 'еталонів працівників', ''],
  ];
  $('#d-tiles').innerHTML = tiles.map(([name, v, sub, cls]) =>
    `<div class="tile ${cls}"><small>${name}</small><b>${v}</b><span>${sub}</span></div>`).join('');
  $('#d-sub').textContent = `події за сьогодні (${d.date}) з журналу інцидентів сервера`;
  ingestEvents(d.incidents);
  const ul = $('#d-incidents');
  ul.replaceChildren(...(d.incidents.length ? d.incidents.map(e => {
    const li = incidentRow(App.incidents.find(x => x.id === e.id) || e);
    return li;
  }) : [Object.assign(document.createElement('li'), { className: 'empty', textContent: 'тривог і перевірок ще не було' })]));
  $('#d-state').innerHTML = [
    ['Камера', d.camera_active ? 'кадри надходять' : 'кадри не надходять'],
    ['Швидкодія', d.fps != null ? `${fmt(d.fps, 1)} кадр/с` : '–'],
    ['Поріг розпізнавання', fmt(d.thresholds.recognition)],
    ['Поріг атаки', fmt(d.thresholds.pad)],
    ['Ключ шифрування', d.key_set ? 'задано' : 'не задано'],
    ['Сервер запущено', fmtDateTime(d.started_at)],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join('');
}

// ----- база осіб -----

async function refreshPersons() {
  let d;
  try { d = await api('/api/persons'); } catch (e) { toast(e.message, 'err'); return; }
  App.persons = d;
  $('#p-count').textContent = d.count;
  $('#p-enc').textContent = d.persons.filter(p => p.stored === 'encrypted' || p.stored === 'locked').length;
  $('#p-mem').textContent = d.persons.filter(p => p.stored === 'memory').length;
  $('#p-keynote').textContent = d.key_set
    ? 'Ключ PAD_DATASET_KEY задано: нові еталони зберігаються зашифрованими (AES-256-GCM).'
    : 'Ключ PAD_DATASET_KEY не задано: нові еталони лишаються в пам\'яті сервера до перезапуску, зашифровані файли не читаються.';
  renderPersonsTable();
}

function renderPersonsTable() {
  const d = App.persons;
  const body = $('#p-table tbody');
  if (!d) return;
  const q = $('#p-search').value.trim().toLowerCase();
  const rows = d.persons.filter(p => !q || p.id.toLowerCase().includes(q));
  body.innerHTML = rows.length ? rows.map(p => `<tr>
      <td class="mono"><b>${esc(p.id)}</b></td><td>${esc(STORED_SHORT[p.stored] || p.stored)}</td>
      <td class="mono">${esc(fmtDateTime(p.registered))}</td>
      <td class="acts"><button class="btn sm" data-p-check="${esc(p.id)}" title="Сеанс перевірки викликами для ${esc(p.id)}">Перевірити</button>
        ${isAdmin() ? `<button class="btn sm" data-p-del="${esc(p.id)}">Видалити</button>` : ''}</td></tr>`).join('')
    : `<tr class="empty"><td colspan="4">${d.persons.length ? 'нічого не знайдено'
      : isAdmin() ? 'база порожня – додайте особу з фото або з камери' : 'база порожня – осіб додає адміністратор'}</td></tr>`;
}

async function deletePerson(pid) {
  const ok = await confirmDialog(`Видалити ${pid}?`,
    `Еталон ${pid} буде видалено з бази (файл data/enrolled/${pid}.npy.enc і пам'ять сервера). Особа більше не розпізнаватиметься. Цю дію не можна скасувати.`);
  if (!ok) return;
  try {
    await api(`/api/persons/${encodeURIComponent(pid)}`, { method: 'DELETE' });
    toast(`Еталон ${pid} видалено`);
  } catch (e) { toast(e.message, 'err'); }
  refreshPersons();
  refreshParticipants();
}

function personId() {
  const id = $('#p-id').value.trim();
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$/.test(id)) {
    toast('Вкажіть псевдонім латиницею (наприклад, P08), без імені та прізвища', 'err');
    return null;
  }
  return id;
}

async function addPersonPhoto(file) {
  const id = personId();
  if (!id) return;
  try {
    const r = await api(`/api/participants/${encodeURIComponent(id)}/image`, { body: file });
    toast(`Особу ${r.person_id} додано з фото: ${STORED_TEXT[r.stored]}`);
  } catch (e) { toast(e.message, 'err'); }
  refreshPersons();
  refreshParticipants();
}

function addPersonCamera() {
  const id = personId();
  if (!id) return;
  if (!Cam.active) { toast('Спершу увімкніть камеру', 'err'); return; }
  if (isActive(App.session)) { toast('Спершу завершіть поточний сеанс', 'err'); return; }
  Stream.send({ cmd: 'enroll', person_id: id });
}

function renderPersons() {
  drawFrame($('#p-canvas'), App.frame, App.frame && App.frame.box ? `обличчя · ${fmt(App.frame.box[4])}` : '');
  $('#p-pv-center').hidden = Cam.active;
  const e = App.enroll;
  $('#p-pv-instr').hidden = !e;
  if (e) {
    $('#p-instr').textContent = `Реєстрація ${e.person_id}: дивіться прямо в камеру`;
    $('#p-pct').textContent = `${Math.round((e.progress || 0) * 100)} %`;
    $('#p-progress').style.width = `${Math.round((e.progress || 0) * 100)}%`;
  }
  $('#p-cam').disabled = !!e;
}

// ----- налаштування -----

async function refreshSettings() {
  let s;
  try { s = await api('/api/settings'); } catch (e) { toast(e.message, 'err'); return; }
  showSettings(s);
}

function showSettings(s) {
  $('#s-rec').value = s.recognition_threshold.toFixed(2);
  $('#s-pad').value = s.pad_threshold.toFixed(2);
  $('#s-rec').readOnly = $('#s-pad').readOnly = !isAdmin();
  $('#s-config').textContent = `У конфігурації: розпізнавання ${fmt(s.config.recognition_threshold)}, атака ${fmt(s.config.pad_threshold)}.`;
  const kv = rows => rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('');
  const file = s.classifier_file ? s.classifier_file.split(/[\\/]/).slice(-2).join('/') : 'немає (базовий поріг s_ref)';
  $('#s-pad-info').innerHTML = kv([
    ['Викликів у сеансі', s.challenge_count], ['Тривалість виклику', `${fmt(s.duration_s, 1)} с`],
    ['Типи викликів', s.challenge_types_short.join(' · ')], ['Повторних спроб', s.max_retries],
    ['Обмеження сеансу', `${fmt(s.session_timeout_s, 0)} с`], ['Файл класифікатора', file],
    ['Версія моделі', s.classifier_version],
  ]);
  $('#s-rec-info').innerHTML = kv([
    ['Детектор', s.detector], ['Ембединг', s.embedder], ['Пошук', s.search],
    ['Кадрів до події', s.confirm_frames], ['Повтор події через', `${fmt(s.event_gap_s, 0)} с`],
  ]);
  $('#s-data-info').innerHTML = kv([
    ['Шифрування', s.encryption.algorithm], ['Ключ PAD_DATASET_KEY', s.encryption.key_set ? 'задано' : 'не задано'],
    ['Сервер', ['127.0.0.1', 'localhost', '[::1]'].includes(location.hostname) ? `${location.host} (лише цей комп'ютер)` : location.host],
    ['З\'єднання', location.protocol === 'https:' ? 'HTTPS і WSS (TLS)' : 'HTTP (без TLS)'],
    ['Вхід', Auth.enabled ? 'JWT, до 24 год, bcrypt' : 'вимкнено'],
  ]);
}

// ----- користувачі і журнал аудиту -----

const ACTION_TEXT = {
  'user.create': 'новий користувач', 'user.update': 'зміна користувача', 'person.add': 'додано особу',
  'person.delete': 'видалено особу', 'settings.update': 'зміна порогів', 'recording.review': 'розмітка запису',
};
const CHANGE_TEXT = {
  role: 'роль', active: 'активний', password: 'пароль', source: 'джерело', stored: 'зберігання',
  deleted_files: 'файлів', reviewed: 'розмічено', recognition_threshold: 'поріг розпізнавання',
  pad_threshold: 'поріг атаки', via: 'через',
};
const changeValue = v => typeof v === 'boolean' ? (v ? 'так' : 'ні') : ROLE_TEXT[v] || (typeof v === 'number' && !Number.isInteger(v) ? fmt(v) : String(v));
const changesText = c => Object.entries(c || {}).map(([k, v]) =>
  `${CHANGE_TEXT[k] || k}: ${Array.isArray(v) ? `${changeValue(v[0])} → ${changeValue(v[1])}` : changeValue(v)}`).join('; ') || '–';

async function refreshUsers() {
  let users, audit;
  try { [users, audit] = await Promise.all([api('/api/users'), api('/api/audit')]); } catch (e) { toast(e.message, 'err'); return; }
  if (App.view !== 'users') return;
  App.users = users;
  renderUsersTable();
  $('#a-table tbody').innerHTML = audit.map(a => `<tr>
      <td class="mono">${esc(fmtDateTime(a.time))}</td><td>${esc(a.user || 'командний рядок')}</td>
      <td>${esc(ACTION_TEXT[a.action] || a.action)}</td><td class="mono">${esc(a.target)}</td><td>${esc(changesText(a.changes))}</td>
    </tr>`).join('') || '<tr class="empty"><td colspan="5">змін ще не було</td></tr>';
}

// Таблиця облікових записів з пошуком за поштою (FR-07).
function renderUsersTable() {
  const me = App.user && App.user.email;
  const q = $('#u-search').value.trim().toLowerCase();
  const users = (App.users || []).filter(u => !q || u.email.includes(q));
  $('#u-table tbody').innerHTML = users.map(u => `<tr data-email="${esc(u.email)}">
      <td><b>${esc(u.email)}</b>${u.email === me ? ' <span class="tag">ви</span>' : ''}</td>
      <td><select class="role" data-u-role aria-label="Роль ${esc(u.email)}">${Object.entries(ROLE_TEXT).map(([r, t]) =>
        `<option value="${r}"${u.role === r ? ' selected' : ''}>${t}</option>`).join('')}</select></td>
      <td>${u.active ? 'активний' : '<span class="state-off">заблоковано</span>'}</td>
      <td class="mono">${esc(fmtDateTime(u.created))}</td>
      <td class="acts"><button class="btn sm" data-u-pw>Пароль</button>${u.email === me ? ''
        : `<button class="btn sm" data-u-active="${u.active ? 0 : 1}">${u.active ? 'Заблокувати' : 'Розблокувати'}</button>`}</td>
    </tr>`).join('') || `<tr class="empty"><td colspan="5">${q ? 'нічого не знайдено' : 'облікових записів немає'}</td></tr>`;
}

async function updateUser(email, body, done) {
  try {
    await api(`/api/users/${encodeURIComponent(email)}`, { method: 'PATCH', json: body });
    toast(done);
  } catch (e) { toast(e.message, 'err'); }
  refreshUsers();
}

async function createUser(ev) {
  ev.preventDefault();
  const body = { email: $('#u-email').value.trim(), password: $('#u-password').value, role: $('#u-role').value };
  try {
    const u = await api('/api/users', { json: body });
    toast(`Створено ${u.email} (${u.role_text})`);
    $('#u-form').reset();
  } catch (e) { toast(e.message, 'err'); }
  refreshUsers();
}

function askPassword(email) {
  $('#pw-title').textContent = `Новий пароль: ${email}`;
  $('#pw-input').value = '';
  $('#pw-modal').hidden = false;
  $('#pw-input').focus();
  $('#pw-form').onsubmit = ev => {
    ev.preventDefault();
    $('#pw-modal').hidden = true;
    updateUser(email, { password: $('#pw-input').value }, `Пароль ${email} змінено`);
  };
}

async function applySettings(reset) {
  const body = reset ? { reset: true } : { recognition_threshold: +$('#s-rec').value, pad_threshold: +$('#s-pad').value };
  if (!reset && [body.recognition_threshold, body.pad_threshold].some(v => !(v >= 0 && v <= 1))) {
    toast('Пороги мають бути в межах від 0 до 1', 'err');
    return;
  }
  try {
    const s = await api('/api/settings', { method: 'PUT', json: body });
    showSettings(s);
    App.config.settings.threshold = s.pad_threshold;
    App.config.monitoring.threshold = s.recognition_threshold;
    toast(reset ? 'Повернуто значення з конфігурації' : 'Пороги змінено до перезапуску сервера');
  } catch (e) { toast(e.message, 'err'); }
}

// ----- маршрути і запуск -----

function render() {
  if (!App.config) return;
  if (App.view === 'verify') renderVerify();
  else if (App.view === 'monitor') renderMonitor();
  else if (App.view === 'record') renderRecord();
  else if (App.view === 'persons') renderPersons();
}

const VIEWS = {
  '#/record': 'record', '#/monitor': 'monitor', '#/dashboard': 'dashboard', '#/persons': 'persons',
  '#/settings': 'settings', '#/users': 'users',
};
const ALL_VIEWS = ['verify', 'monitor', 'record', 'review', 'dashboard', 'persons', 'settings', 'users'];
const ADMIN_VIEWS = new Set(['record', 'users']);

function route() {
  const prev = App.view;
  const rv = location.hash.match(/^#\/review\/([^/]+)$/);
  App.view = rv ? 'review' : VIEWS[location.hash] || 'verify';
  // оператор, що відкрив адресу вкладки адміністратора, потрапляє на верифікацію
  if (ADMIN_VIEWS.has(App.view) && !isAdmin()) App.view = 'verify';
  for (const v of ALL_VIEWS) $(`#view-${v}`).hidden = App.view !== v;
  // на вузькому екрані сторінка прокручується цілком: нова вкладка відкривається з початку
  if (prev !== App.view) window.scrollTo(0, 0);
  if (App.view === 'dashboard') refreshDashboard();
  if (App.view === 'persons') refreshPersons();
  if (App.view === 'settings') refreshSettings();
  if (App.view === 'users') refreshUsers();
  // перегляд сеансу підсвічує вкладку, з якої його відкрито
  const tab = App.view === 'review' ? VIEWS[App.lastHash] || 'verify' : App.view;
  $$('.topbar nav a[data-view]').forEach(a => a.classList.toggle('active', a.dataset.view === tab));
  closeIncident();
  if (App.view === 'review') {
    $('#rv-back').href = App.lastHash;
    const uid = decodeURIComponent(rv[1]);
    if (Review.uid !== uid) Review.open(uid);
  } else {
    App.lastHash = location.hash || '#/verify';
    if (prev === 'review') Review.close();
  }
  schedule();
  // моніторинг використовує той самий потік кадрів: сервер перемикає режим з'єднання
  if (App.view === 'monitor' && prev !== 'monitor') {
    facesKey = null;
    App.frame = null;
    Stream.send({ cmd: 'monitor', on: true });
  } else if (prev === 'monitor' && App.view !== 'monitor') {
    App.frame = null;
    Stream.send({ cmd: 'monitor', on: false });
  }
  if (App.view === 'record') refreshRecordings();
  render();
}

function fillForm() {
  const cfg = App.config;
  $('#r-protocol').textContent = `Протокол збору ${cfg.protocol}`;
  $('#r-lighting').innerHTML = Object.entries(cfg.lighting).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join('');
  $('#r-tool').innerHTML = Object.entries(cfg.swap_tools).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join('');
  $('#r-count').innerHTML = [2, 3, 4, 5].map(n => `<option value="${n}">${n} (випадково)</option>`).join('');
  $('#r-count').value = String(cfg.settings.challenge_count);
  $('#r-duration').innerHTML = [3, 4, 5].map(n => `<option value="${n}">${n} с</option>`).join('');
  $('#r-duration').value = String(Math.round(cfg.settings.duration_s));
  $('#r-types').innerHTML = cfg.challenge_types.map(t =>
    `<label><input type="checkbox" value="${t.id}" ${cfg.settings.challenge_types.includes(t.id) ? 'checked' : ''}> ${esc(t.label)}</label>`).join('');
}

function bind() {
  $$('.cam-on').forEach(b => b.addEventListener('click', () => Cam.start()));
  $$('.video-file').forEach(i => i.addEventListener('change', () => i.files[0] && Cam.startFile(i.files[0])));
  $$('.source-select').forEach(sel => sel.addEventListener('change', () => {
    if (sel.value === 'pick-file') { sel.value = Cam.deviceId || ''; pickFile(); } else if (sel.value && sel.value !== 'file') Cam.start(sel.value);
  }));
  $('#v-person').addEventListener('input', renderPersonInfo);
  $('#m-reload').addEventListener('click', () => Stream.send({ cmd: 'monitor', on: true }));
  $('#m-faces').addEventListener('click', e => {
    const b = e.target.closest('[data-check]');
    if (b) checkPerson(b.dataset.check);
  });
  $('#v-enroll').addEventListener('click', startEnroll);
  $('#v-photo').addEventListener('change', e => { if (e.target.files[0]) enrollPhoto(e.target.files[0]); e.target.value = ''; });
  $('#v-main').addEventListener('click', () => (isActive(App.session) ? cancelSession() : startVerify()));
  $('#v-repeat').addEventListener('click', async () => { await cancelSession(); startVerify(); });
  $('#r-start').addEventListener('click', () => {
    if (isActive(App.session)) { cancelSession(); return; }
    startSession({ mode: 'record', recording: recordParams() });
  });
  $('.form').addEventListener('input', render);
  $('#r-types').addEventListener('change', render);
  $('#r-blur').addEventListener('change', render);
  $('#r-table').addEventListener('click', e => {
    const b = e.target.closest('[data-review]');
    if (b) review(b.dataset.review, b.dataset.to === '1');
    const v = e.target.closest('[data-open-review]');
    if (v) location.hash = `#/review/${encodeURIComponent(v.dataset.openReview)}`;
  });

  // дашборд, база осіб, налаштування
  $('#d-incidents').addEventListener('click', e => {
    const li = e.target.closest('[data-inc]');
    if (li) openIncident(+li.dataset.inc);
  });
  setInterval(() => { if (App.view === 'dashboard') refreshDashboard(); }, 3000);
  $('#p-search').addEventListener('input', renderPersonsTable);
  $('#p-table').addEventListener('click', e => {
    const del = e.target.closest('[data-p-del]');
    if (del) deletePerson(del.dataset.pDel);
    const chk = e.target.closest('[data-p-check]');
    if (chk) checkPerson(chk.dataset.pCheck);
  });
  $('#p-photo').addEventListener('change', e => { if (e.target.files[0]) addPersonPhoto(e.target.files[0]); e.target.value = ''; });
  $('#p-cam').addEventListener('click', addPersonCamera);
  $('#s-apply').addEventListener('click', () => applySettings(false));
  $('#s-reset').addEventListener('click', () => applySettings(true));

  // користувачі
  $('#logout').addEventListener('click', () => Auth.logout());
  $('#u-form').addEventListener('submit', createUser);
  $('#u-search').addEventListener('input', renderUsersTable);
  $('#u-table').addEventListener('change', e => {
    const sel = e.target.closest('[data-u-role]');
    if (!sel) return;
    const email = sel.closest('tr').dataset.email;
    updateUser(email, { role: sel.value }, `${email}: роль – ${ROLE_TEXT[sel.value]}`);
  });
  $('#u-table').addEventListener('click', async e => {
    const row = e.target.closest('tr[data-email]');
    if (!row) return;
    const email = row.dataset.email;
    if (e.target.closest('[data-u-pw]')) askPassword(email);
    const act = e.target.closest('[data-u-active]');
    if (!act) return;
    const active = act.dataset.uActive === '1';
    if (!active && !await confirmDialog(`Заблокувати ${email}?`,
      'Користувач не зможе увійти, його відкриті сесії завершаться одразу. Розблокувати можна тут же.', 'Заблокувати')) return;
    updateUser(email, { active }, active ? `${email} розблоковано` : `${email} заблоковано`);
  });
  $('#pw-cancel').addEventListener('click', () => { $('#pw-modal').hidden = true; });

  // сповіщення і журнал інцидентів
  $('#sound').addEventListener('click', () => Sound.toggle());
  $('#m-only-alarms').addEventListener('change', renderIncidents);
  const openInc = e => {
    const li = e.target.closest('[data-inc]');
    if (li) openIncident(+li.dataset.inc);
  };
  $('#m-incidents').addEventListener('click', openInc);
  $('#m-incidents').addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openInc(e); } });
  $('#inc-close').addEventListener('click', closeIncident);
  $('#incident').addEventListener('click', e => {
    if (e.target.id === 'incident') closeIncident();
    const b = e.target.closest('[data-review-open]');
    if (b) location.hash = `#/review/${encodeURIComponent(b.dataset.reviewOpen)}`;
  });

  // перегляд сеансу
  $('#rv-play').addEventListener('click', () => Review.toggle());
  $('#rv-scrub').addEventListener('input', e => { Review.pause(); Review.seek(+e.target.value / 1000 * Review.end); });
  $$('.speed [data-speed]').forEach(b => b.addEventListener('click', () => Review.setSpeed(+b.dataset.speed)));
  const chart = $('#rv-chart');
  const seekChart = e => {
    const map = Review.chartMap;
    if (!map || !Review.meta) return;
    const r = chart.getBoundingClientRect();
    Review.seek(map.tMin + (e.clientX - r.left - map.L) / (map.R - map.L) * (map.tMax - map.tMin));
  };
  chart.addEventListener('pointerdown', e => { Review.pause(); seekChart(e); chart.setPointerCapture(e.pointerId); });
  chart.addEventListener('pointermove', e => { if (e.buttons & 1) seekChart(e); });
  $('#rv-challenges').addEventListener('click', e => {
    const li = e.target.closest('li[data-i]');
    if (!li || !Review.meta) return;
    const c = Review.meta.challenges[+li.dataset.i];
    if (c.t_start != null) { Review.pause(); Review.seek(c.t_start - Review.meta.settings.before_s); }
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !$('#incident').hidden) closeIncident();
    if (e.key === 'Escape' && !$('#confirm').hidden) $('#cf-no').click();
    if (e.key === 'Escape' && !$('#pw-modal').hidden) $('#pw-modal').hidden = true;
    if (App.view !== 'review' || e.target.closest('input, button, select, textarea')) return;
    if (e.key === ' ') { e.preventDefault(); Review.toggle(); }
    if (e.key === 'ArrowRight') Review.seek(Review.tau + 0.5);
    if (e.key === 'ArrowLeft') Review.seek(Review.tau - 0.5);
  });

  window.addEventListener('hashchange', route);
  window.addEventListener('resize', () => { render(); if (App.view === 'review') Review.update(); });
}

async function init() {
  Cam.init();
  $('#login-form').addEventListener('submit', e => Auth.submit(e));
  let me;
  try {
    me = await api('/api/auth/me');
  } catch (e) {
    Auth.showLogin(e.status === 401 ? '' : `Сервер недоступний: ${e.message}`);
    return;
  }
  Auth.enabled = me.auth;
  Auth.apply(me.user);
  await start();
}

// Інтерфейс після входу: конфігурація модуля, форми, журнал, потік кадрів.
async function start() {
  App.started = true;
  try {
    App.config = await api('/api/config');
  } catch (e) {
    App.started = false;
    toast(`Сервер недоступний: ${e.message}`, 'err');
    return;
  }
  fillForm();
  Sound.init();
  bind();
  await fillDevices();
  await refreshParticipants();
  await refreshIncidents(true);   // події до відкриття сторінки – у журналі, але без сповіщень
  route();
  Stream.connect();
}

init();
