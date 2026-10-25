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

const App = {
  config: null, participants: [], view: 'verify',
  session: null, frame: null, enroll: null, recordings: null, monitor: null,
};

// ----- службові -----

async function api(path, opts = {}) {
  const headers = opts.json ? { 'Content-Type': 'application/json' } : {};
  const r = await fetch(path, { method: opts.method || (opts.json || opts.body ? 'POST' : 'GET'), headers,
    body: opts.json ? JSON.stringify(opts.json) : opts.body });
  let data = null;
  try { data = await r.json(); } catch (_) { /* тіло без JSON */ }
  if (!r.ok) {
    const d = data && data.detail;
    throw new Error(typeof d === 'string' ? d : Array.isArray(d) ? d.map(x => x.msg).join('; ') : `HTTP ${r.status}`);
  }
  return data;
}

function toast(msg, kind = '') {
  const el = $('#toast');
  el.textContent = msg;
  el.className = `toast show ${kind}`;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.className = 'toast'; }, 4500);
}

// ----- камера -----

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
      toast(`Камера недоступна: ${e.message}`, 'err');
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
    await this.video.play();
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
      if (App.view === 'monitor') this.send({ cmd: 'monitor', on: true });
      schedule();
    };
    ws.onclose = () => {
      this.open = false;
      this.inflight = false;
      setLive();
      // сервер перериває сеанс, якщо з'єднання закрито
      if (isActive(App.session)) refreshSession();
      setTimeout(() => this.connect(), 1500);
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
  if (!Cam.active || !Stream.open || Stream.inflight) return;
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
    const back = s.origin === 'monitor' ? ' · результат записано в журнал моніторингу' : '';
    toast(`Рішення: ${DECISION_TEXT[s.decision.decision].toLowerCase()}${back}`);
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
  renderPersonInfo();
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
  if (m.hand) {
    const p = i => [X(m.hand[2 * i]), m.hand[2 * i + 1]];
    const bones = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
      [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = Math.max(2, 3 * k);
    ctx.beginPath();
    for (const [a, b] of bones) { ctx.moveTo(...p(a)); ctx.lineTo(...p(b)); }
    ctx.stroke();
    ctx.fillStyle = '#1c1c1c';
    for (let i = 0; i < 21; i++) {
      ctx.beginPath();
      ctx.arc(...p(i), Math.max(3, 6 * k), 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
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
  if (!s) return;
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
  $('#c-hand').textContent = c && c.hand_points ? `${c.hand_points} т.` : '–';
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

let facesKey = null, logKey = null;

function eventCells(e) {
  if (e.kind === 'pad') {
    const text = DECISION_TEXT[e.decision] ? DECISION_TEXT[e.decision].toLowerCase() : e.decision;
    return `<td>${esc(e.person_id)} · перевірка: <b>${esc(text)}</b></td><td>${e.score != null ? 'оцінка ' + fmt(e.score) : '–'}</td>`;
  }
  const who = e.kind === 'known' ? esc(e.person_id) : '<b class="unknown">невідома</b>';
  return `<td>${who}</td><td>${fmt(e.similarity)}</td>`;
}

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

  // журнал подій
  const events = mon ? mon.events : (st ? st.events.slice(0, 12) : []);
  const lk = events.length ? `${events.length}|${events[0].time}|${events[0].kind}|${events[0].person_id}` : '';
  if (lk !== logKey) {
    logKey = lk;
    $('#m-log tbody').innerHTML = events.length
      ? events.map(e => `<tr class="${e.kind}"><td class="mono">${esc(e.time)}</td>${eventCells(e)}</tr>`).join('')
      : '<tr class="empty"><td colspan="3">подій ще немає</td></tr>';
  }

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
    body.innerHTML = `<tr class="empty"><td colspan="9">${esc(e.message)}</td></tr>`;
    return;
  }
  const { rows, summary } = App.recordings;
  const tools = App.config.swap_tools;
  body.innerHTML = rows.length ? rows.map(r => {
    if (!r.presentation_type) {
      return `<tr><td class="mono">${r.id}</td><td>${esc(r.participant)}</td><td colspan="6" class="muted">${esc(r.name)}</td><td>${statusCell(r)}</td></tr>`;
    }
    const attack = r.presentation_type === 'attack';
    return `<tr title="${esc(r.name)}${r.score != null ? ' · оцінка ' + fmt(r.score) : ''}">
      <td class="mono">${r.id}</td><td>${esc(r.participant)}</td>
      <td><span class="badge ${attack ? 'attack' : ''}">${attack ? 'Атака' : 'Справжнє'}</span></td>
      <td>${attack ? esc(tools[r.swap_tool] || r.swap_tool) : '–'}</td>
      <td>${esc(r.target_face) || '–'}</td>
      <td>${r.challenges.map(esc).join(' · ') || '–'}</td>
      <td>${conditionText(r)}</td>
      <td class="mono">${r.duration_s != null ? fmt(r.duration_s, 1) + ' с' : '–'}</td>
      <td>${statusCell(r)}</td></tr>`;
  }).join('') : '<tr class="empty"><td colspan="9">Записів ще немає</td></tr>';
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

// ----- маршрути і запуск -----

function render() {
  if (!App.config) return;
  if (App.view === 'verify') renderVerify();
  else if (App.view === 'monitor') renderMonitor();
  else renderRecord();
}

const VIEWS = { '#/record': 'record', '#/monitor': 'monitor' };

function route() {
  const prev = App.view;
  App.view = VIEWS[location.hash] || 'verify';
  for (const v of ['verify', 'monitor', 'record']) $(`#view-${v}`).hidden = App.view !== v;
  $$('.topbar nav a[data-view]').forEach(a => a.classList.toggle('active', a.dataset.view === App.view));
  // моніторинг використовує той самий потік кадрів: сервер перемикає режим з'єднання
  if (App.view === 'monitor' && prev !== 'monitor') {
    facesKey = logKey = null;
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
  });
  window.addEventListener('hashchange', route);
  window.addEventListener('resize', render);
}

async function init() {
  Cam.init();
  try {
    App.config = await api('/api/config');
  } catch (e) {
    toast(`Сервер недоступний: ${e.message}`, 'err');
    return;
  }
  fillForm();
  bind();
  await fillDevices();
  await refreshParticipants();
  route();
  Stream.connect();
}

init();
