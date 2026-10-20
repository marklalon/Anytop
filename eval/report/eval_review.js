import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { BVHLoader } from 'three/addons/BVHLoader.js';
import { GLTFLoader } from 'three/addons/GLTFLoader.js';

const report = JSON.parse(document.getElementById('report-data').textContent);
const $ = id => document.getElementById(id);
const el = (tag, text, className) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
};
const color = score => score == null ? 'var(--edge)' :
  score <= 2 ? 'var(--red)' : score <= 4 ? 'var(--orange)' : score <= 6 ? 'var(--yellow)' : 'var(--green)';
const fmt = value => value == null ? '—' : Number(value).toFixed(4);
let current = null, views = [], ratings = {}, connected = false, saving = false;
let playing = true, seconds = 0, duration = 0, frameTime = 1 / 30, generation = 0;
let loading = false, loadController = null, lastNow = performance.now();
const taskNodes = new Map();
const stage = el('div');
stage.style.cssText = 'position:absolute;pointer-events:none;overflow:hidden;z-index:0';
$('center').style.position = 'relative';
$('center').insertBefore(stage, $('views-scroll'));
$('views-scroll').style.zIndex = '1';

// A shared WebGL context supports arbitrarily many scrollable views without
// exhausting the browser's limit on simultaneous WebGL contexts.
let renderer = null;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x202020);
  stage.append(renderer.domElement);
} catch (error) {
  $('connection').textContent = `无法初始化 3D 预览：${error.message}`;
  $('connection').classList.add('error');
}

$('report-name').textContent = [report.run_name, report.model_name, report.task_config_name].filter(Boolean).join(' / ');
$('report-name').title = $('report-name').textContent;

function visibleTasks() {
  const words = $('search').value.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
  return report.tasks.filter(task => words.every(word => `${task.id} ${task.command}`.toLocaleLowerCase().includes(word)));
}

function updateTaskBadge(task) {
  const node = taskNodes.get(task.id);
  const score = ratings[task.id]?.score;
  node.style.setProperty('--rating-color', color(score));
  node.querySelector('.badge').textContent = score ?? '—';
  node.setAttribute('aria-label', `${task.id}，${score == null ? '未评分' : score + ' 分'}，${task.command}`);
}

for (const task of report.tasks) {
  const node = el('button', undefined, 'task');
  const heading = el('div', undefined, 'task-heading');
  heading.append(el('span', task.id, 'task-name'), el('span', '—', 'badge'));
  node.append(heading, el('pre', task.command || '（无 command）'));
  node.addEventListener('click', () => selectTask(task));
  taskNodes.set(task.id, node);
  updateTaskBadge(task);
  $('tasks').append(node);
}

function filterTasks() {
  const visible = visibleTasks();
  const ids = new Set(visible.map(task => task.id));
  for (const [id, node] of taskNodes) node.hidden = !ids.has(id);
  $('task-count').textContent = `${visible.length} / ${report.tasks.length} tasks`;
  if (!current || !ids.has(current.id)) selectTask(visible[0] || null);
}
$('search').addEventListener('input', filterTasks);

function updateRatingsPanel() {
  const score = current && ratings[current.id]?.score;
  $('current-rating').textContent = score == null ? '未评分' : `${score} / 9`;
  for (const button of $('rating-buttons').children) {
    button.disabled = !current || !connected || saving;
    const selected = score === Number(button.dataset.score);
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  }
  $('clear-rating').disabled = score == null || !connected || saving;
  $('rated-count').textContent = `已评分 ${Object.keys(ratings).length} / ${report.tasks.length}`;
}

// Lay out the buttons like a numpad: 7-8-9 / 4-5-6 / 1-2-3.
for (const score of [7, 8, 9, 4, 5, 6, 1, 2, 3]) {
  const button = el('button', String(score), 'rating');
  button.dataset.score = score;
  button.style.setProperty('--rating-color', color(score));
  button.addEventListener('click', () => saveRating(score));
  $('rating-buttons').append(button);
}
$('clear-rating').addEventListener('click', () => saveRating(null));

async function jsonRequest(url, options) {
  const response = await fetch(url, { cache: 'no-store', ...options });
  const type = response.headers.get('Content-Type') || '';
  if (!type.includes('application/json')) throw new Error('请使用报告自带的 serve_report.py 启动评分服务');
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

async function connect() {
  if (location.protocol === 'file:') return;
  try {
    const data = await jsonRequest('./api/ratings');
    ratings = data.ratings;
    connected = true;
    if (renderer) {
      $('connection').textContent = data.stale_count ? `${data.stale_count} 条旧评分不适用于当前任务，请重新评审对应任务。` : '';
      $('connection').classList.remove('error');
    }
    $('save-status').textContent = '评分自动保存到 human_scores.json';
    $('save-status').classList.remove('error');
    for (const task of report.tasks) updateTaskBadge(task);
  } catch (error) {
    connected = false;
    $('connection').replaceChildren(el('span', `评分服务连接失败：${error.message} `));
    const retry = el('button', '重新连接');
    retry.addEventListener('click', connect);
    $('connection').append(retry);
    $('connection').classList.add('error');
    $('save-status').textContent = '未连接，无法保存评分';
  }
  updateRatingsPanel();
}

async function saveRating(score) {
  if (!current || !connected || saving) return;
  const task = current;
  saving = true;
  updateRatingsPanel();
  $('save-status').textContent = '正在保存…';
  $('save-status').classList.remove('error');
  try {
    const data = await jsonRequest('./api/ratings', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_id: task.id, revision: task.revision, score }),
    });
    if (data.rating) ratings[task.id] = data.rating;
    else delete ratings[task.id];
    updateTaskBadge(task);
    $('save-status').textContent = `${task.id}：${score == null ? '已清除评分' : score + ' 分已保存'}`;
  } catch (error) {
    // Keep the last confirmed rating visible if saving fails.
    $('save-status').textContent = `保存失败：${error.message}。可重试评分。`;
    $('save-status').classList.add('error');
  } finally {
    saving = false;
    updateRatingsPanel();
  }
}

// Same BVH conventions as dataset/review/joint_names.html: position channels
// replace offsets; direct track application preserves names with dots/colons.
function parseBVH(text) {
  const parsed = new BVHLoader().parse(text);
  const bones = parsed.skeleton.bones.filter(bone => bone.name !== 'ENDSITE');
  const hasPosition = [];
  for (const line of text.split(/^MOTION/m)[0].split('\n')) {
    const value = line.trim();
    if (/^(ROOT|JOINT)\s/.test(value)) hasPosition.push(false);
    else if (value.startsWith('CHANNELS') && /position/i.test(value)) hasPosition[hasPosition.length - 1] = true;
  }
  const tracks = parsed.clip.tracks;
  const frames = tracks[0]?.times.length || 0;
  const frameTime = Number(/^Frame Time:\s*([\d.eE+-]+)/m.exec(text)?.[1]);
  if (!bones.length || !frames || !Number.isFinite(frameTime) || frameTime <= 0) throw new Error('BVH 没有有效的帧数据');
  const pos = bones.map((bone, index) => {
    const values = Float32Array.from(tracks[index * 2].values);
    if (hasPosition[index]) {
      for (let frame = 0; frame < frames; frame++) {
        values[3 * frame] -= bone.position.x;
        values[3 * frame + 1] -= bone.position.y;
        values[3 * frame + 2] -= bone.position.z;
      }
    }
    return values;
  });
  return { root: parsed.skeleton.bones[0], bones, pos,
    quat: bones.map((bone, index) => tracks[index * 2 + 1].values), frames, frameTime,
    duration: Math.max(0, frames - 1) * frameTime };
}

const vec = new THREE.Vector3(), quat = new THREE.Quaternion(), matrix = new THREE.Matrix4();
function applyPose(bvh, time) {
  if (bvh.mixer) {
    if (bvh.action) bvh.action.paused = false;
    bvh.mixer.setTime(Math.min(Math.max(time, 0), bvh.duration));
    bvh.root.updateMatrixWorld(true);
    return;
  }
  const frame = Math.min(Math.max(time / bvh.frameTime, 0), bvh.frames - 1);
  const lo = Math.floor(frame), hi = Math.min(lo + 1, bvh.frames - 1), weight = frame - lo;
  bvh.bones.forEach((bone, index) => {
    bone.position.fromArray(bvh.pos[index], lo * 3).lerp(vec.fromArray(bvh.pos[index], hi * 3), weight);
    bone.quaternion.fromArray(bvh.quat[index], lo * 4).slerp(quat.fromArray(bvh.quat[index], hi * 4), weight);
  });
  bvh.root.updateMatrixWorld(true);
}

async function parseReference(buffer, url) {
  if (!/\.glb$/i.test(url)) return parseBVH(new TextDecoder().decode(buffer));
  const gltf = await new GLTFLoader().parseAsync(buffer, '');
  const root = gltf.scene, bones = [];
  root.traverse(object => {
    if (object.isBone) bones.push(object);
    // Render the reference skeleton only, including for skinned GLB files.
    if (object.isMesh) object.visible = false;
  });
  if (!bones.length) throw new Error('GLB 没有骨架');
  const clip = gltf.animations[0];
  const duration = clip?.duration || 0;
  const mixer = new THREE.AnimationMixer(root);
  let action = null;
  if (clip) {
    action = mixer.clipAction(clip);
    action.setLoop(THREE.LoopOnce, 1);
    action.clampWhenFinished = true;
    action.play();
  }
  const times = clip?.tracks.flatMap(track => Array.from(track.times)) || [];
  const uniqueTimes = [...new Set(times)].sort((a, b) => a - b);
  const frameTime = uniqueTimes.length > 1 ? duration / (uniqueTimes.length - 1) : 1 / 30;
  return { root, bones, mixer, action, duration, frameTime, frames: Math.max(1, uniqueTimes.length) };
}

function resetCamera(view, kind = 'persp') {
  if (!view.bvh) return;
  const offset = { persp: [1.4, .8, 1.8], front: [0, 0, 2.5], side: [2.5, 0, 0], top: [0, 2.5, .0001] }[kind];
  const aspect = Math.max(.1, view.node.clientWidth / Math.max(1, view.node.clientHeight));
  const distance = view.extent * Math.max(1, 1 / aspect);
  view.camera.position.copy(view.center).add(new THREE.Vector3(...offset).multiplyScalar(distance));
  view.camera.near = view.extent / 1000;
  view.camera.far = view.extent * 1000;
  view.camera.updateProjectionMatrix();
  view.controls.target.copy(view.center);
  // Remember the framing, so user camera moves can be mirrored onto other views.
  view.homePosition = view.camera.position.clone();
  view.homeTarget = view.controls.target.clone();
  view.controls.update();
}

// Mirror a view's camera onto every other view. Each view keeps its own
// framing scale, so clips whose motion spans a different area stay visible
// while sharing the exact orientation and relative zoom.
let cameraSource = null;
const syncPosition = new THREE.Vector3(), syncTarget = new THREE.Vector3();
function syncCameras(source) {
  if (!source.homePosition || !source.homeTarget) return;
  cameraSource = source;
  syncPosition.copy(source.camera.position).sub(source.homePosition);
  syncTarget.copy(source.controls.target).sub(source.homeTarget);
  for (const view of views) {
    if (view === source || !view.homePosition || !view.homeTarget) continue;
    const scale = view.extent / source.extent;
    view.camera.position.copy(view.homePosition).addScaledVector(syncPosition, scale);
    view.controls.target.copy(view.homeTarget).addScaledVector(syncTarget, scale);
    view.camera.lookAt(view.controls.target);
    view.camera.updateMatrixWorld();
  }
}

function setupSkeleton(view, bvh, scene, reference = false) {
  view.bvh = bvh;
  scene.add(bvh.root);
  view.positions = bvh.bones.map(() => new THREE.Vector3());
  // Fit the whole trajectory, so forward motion stays in view throughout a loop.
  const box = new THREE.Box3();
  const samples = Math.min(bvh.frames, 160);
  for (let index = 0; index < samples; index++) {
    applyPose(bvh, samples <= 1 ? 0 : bvh.duration * index / (samples - 1));
    bvh.bones.forEach((bone, i) => box.expandByPoint(bone.getWorldPosition(view.positions[i])));
  }
  view.center = box.getCenter(new THREE.Vector3());
  view.extent = Math.max(...box.getSize(new THREE.Vector3()).toArray(), .001);
  applyPose(bvh, 0);
  const bodyBox = new THREE.Box3().setFromPoints(bvh.bones.map(bone => bone.getWorldPosition(new THREE.Vector3())));
  view.radius = Math.max(bodyBox.getSize(new THREE.Vector3()).length() * .009, view.extent * .001);
  const ghost = reference ? { transparent: true, opacity: .4, depthWrite: false } : {};
  view.joints = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 8, 6), new THREE.MeshBasicMaterial({ color: reference ? 0xaaaaaa : 0x85baff, ...ghost }), bvh.bones.length);
  view.joints.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  view.joints.frustumCulled = false;
  scene.add(view.joints);
  const indices = new Map(bvh.bones.map((bone, i) => [bone, i]));
  view.segments = bvh.bones.map((bone, i) => [i, indices.get(bone.parent)]).filter(pair => pair[1] !== undefined);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(view.segments.length * 6, 3));
  view.lines = new THREE.LineSegments(geometry, new THREE.LineBasicMaterial({ color: reference ? 0xaaaaaa : 0xaacfff, ...ghost }));
  view.lines.frustumCulled = false;
  scene.add(view.lines);
}

function setupView(view, text, reference) {
  const scene = new THREE.Scene();
  view.scene = scene;
  setupSkeleton(view, parseBVH(text), scene);
  if (reference) {
    view.reference = {};
    setupSkeleton(view.reference, reference, scene, true);
    // Frame both trajectories without changing their relative positions.
    const box = new THREE.Box3();
    for (const skeleton of [view, view.reference]) {
      const half = new THREE.Vector3().setScalar(skeleton.extent / 2);
      box.expandByPoint(skeleton.center.clone().sub(half));
      box.expandByPoint(skeleton.center.clone().add(half));
    }
    view.center = box.getCenter(new THREE.Vector3());
    view.extent = Math.max(...box.getSize(new THREE.Vector3()).toArray(), .001);
  }
  const grid = new THREE.GridHelper(view.extent * 4, 40, 0x555555, 0x363636);
  grid.position.set(view.center.x, 0, view.center.z);
  scene.add(grid);
  view.camera = new THREE.PerspectiveCamera(40, 1, .001, 10000);
  view.controls = new OrbitControls(view.camera, view.node);
  // Match dataset/review/joint_names.html, including modifier-key panning.
  view.controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.ROTATE, RIGHT: THREE.MOUSE.PAN };
  view.controls.enableDamping = false;
  view.controls.addEventListener('change', () => syncCameras(view));
  view.message.hidden = true;
  const previousSource = cameraSource;
  resetCamera(view);
  // A clip that finishes loading later adopts the camera the user already moved.
  if (previousSource && previousSource !== view && previousSource.homePosition) syncCameras(previousSource);
}

function disposeViews() {
  for (const view of views) {
    view.controls?.dispose();
    if (view.reference?.bvh.mixer) {
      view.reference.bvh.mixer.stopAllAction();
      view.reference.bvh.mixer.uncacheRoot(view.reference.bvh.root);
    }
    view.scene?.traverse(object => {
      object.geometry?.dispose();
      if (object.material) for (const material of [].concat(object.material)) material.dispose();
      if (object.isInstancedMesh) object.dispose();
    });
  }
  renderer?.renderLists.dispose();
  views = [];
  cameraSource = null;
  $('views').replaceChildren();
}

async function selectTask(task) {
  if (task === current && task !== null) return;
  loadController?.abort();
  loadController = new AbortController();
  const signal = loadController.signal;
  const token = ++generation;
  current = task;
  for (const [id, node] of taskNodes) {
    const selected = id === task?.id;
    node.classList.toggle('active', selected);
    node.setAttribute('aria-current', String(selected));
  }
  disposeViews();
  seconds = 0; duration = 0; loading = true;
  $('task-title').textContent = task?.id || '没有匹配的任务';
  $('clip-count').textContent = task ? `${task.clips.length} 个动作${task.reference_clip ? ' · 灰色半透明：参考动作' : ''}` : '';
  $('frame-info').textContent = '';
  $('quality').textContent = fmt(task?.quality);
  $('dimensions').textContent = Object.entries(task?.dimensions || {}).map(([key, value]) => `${key} ${fmt(value)}`).join(' · ');
  $('task-status').textContent = task?.status || '';
  $('score-label').textContent = task?.score_label ? `评估标签：${task.score_label}` : '';
  $('command').textContent = task?.command || '';
  if (!saving && connected) {
    $('save-status').textContent = '评分自动保存到 human_scores.json';
    $('save-status').classList.remove('error');
  }
  updateRatingsPanel();
  updateTransport();
  if (!task || !task.clips.length) {
    $('views').append(el('div', task ? '此任务没有可预览的 BVH 动作文件。' : '尝试其他任务名或参数。', 'empty'));
    loading = false;
    return;
  }
  // Fetch once, then parse separate skeletons for independent scenes.
  const referenceData = task.reference_clip ? fetch(task.reference_clip.url, { signal }).then(response => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.arrayBuffer();
  }).then(buffer => ({ buffer }), error => ({ error })) : Promise.resolve(null);
  const jobs = task.clips.map(clip => {
    const node = el('div', undefined, 'view');
    node.setAttribute('aria-label', clip.name);
    const message = el('div', renderer ? '正在加载…' : 'WebGL 不可用', 'view-message');
    node.append(message);
    $('views').append(node);
    const view = { node, message };
    views.push(view);
    if (!renderer) return Promise.resolve();
    return fetch(clip.url, { signal }).then(response => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.text();
    }).then(async text => {
      const data = await referenceData;
      if (token !== generation) return;
      let reference = null, referenceError = data?.error;
      if (data?.buffer) {
        try { reference = await parseReference(data.buffer, task.reference_clip.url); }
        catch (error) { referenceError = error; }
      }
      if (token !== generation) return;
      setupView(view, text, reference);
      if (referenceError) {
        view.message.hidden = false;
        view.message.textContent = `参考动作加载失败：${referenceError.message}`;
        view.message.classList.add('error');
      }
    }).catch(error => {
      if (token !== generation) return;
      view.message.hidden = false;
      view.message.textContent = `加载失败：${error.message}`;
      view.message.classList.add('error');
    });
  });
  await Promise.allSettled(jobs);
  if (token !== generation) return;
  const firstClip = views.find(view => view.bvh)?.bvh;
  $('frame-info').textContent = firstClip ? `${firstClip.frames} 帧 · ${(1 / firstClip.frameTime).toFixed(1)} FPS` : '';
  const motions = views.flatMap(view => [view.bvh, view.reference?.bvh]).filter(Boolean);
  duration = Math.max(0, ...motions.map(motion => motion.duration));
  const frameTimes = motions.map(motion => motion.frameTime);
  frameTime = frameTimes.length ? Math.min(...frameTimes) : 1 / 30;
  loading = false;
  lastNow = performance.now();
  updateTransport();
}

function updateTransport() {
  const enabled = !loading && views.some(view => view.bvh);
  for (const id of ['play', 'step-back', 'step-forward', 'scrub']) $(id).disabled = !enabled;
  $('play').textContent = playing ? '暂停' : '播放';
  $('scrub').max = duration || 1;
  $('scrub').value = seconds;
  $('time').textContent = `${seconds.toFixed(2)} / ${duration.toFixed(2)} s`;
}
$('play').addEventListener('click', () => { playing = !playing; lastNow = performance.now(); updateTransport(); });
$('scrub').addEventListener('input', () => { playing = false; seconds = Number($('scrub').value); updateTransport(); });
function step(direction) {
  playing = false;
  seconds = Math.max(0, Math.min(duration, seconds + direction * frameTime));
  updateTransport();
}
$('step-back').addEventListener('click', () => step(-1));
$('step-forward').addEventListener('click', () => step(1));
document.addEventListener('keydown', event => {
  if ($('summary').open || event.ctrlKey || event.metaKey || event.altKey) return;
  const editing = event.target.closest('input:not([type="range"]),select,textarea') || event.target.isContentEditable;
  // Use physical numpad codes so rating also works with Num Lock turned off,
  // when e.g. Numpad2 reports its key as ArrowDown.
  if (/^Numpad[1-9]$/.test(event.code)) {
    if (!editing) {
      event.preventDefault();
      if (!event.repeat) saveRating(Number(event.code.slice(-1)));
    }
    return;
  }
  if ((event.key === 'ArrowUp' || event.key === 'ArrowDown') && (!editing || event.target === $('search'))) {
    event.preventDefault();
    const tasks = visibleTasks(), index = tasks.indexOf(current);
    const task = tasks[index + (event.key === 'ArrowDown' ? 1 : -1)];
    if (task) { selectTask(task); taskNodes.get(task.id).scrollIntoView({ block: 'nearest' }); }
    return;
  }
  if (editing) return;
  if (event.code === 'Space' && !event.target.closest('button') && !$('play').disabled) {
    event.preventDefault(); $('play').click();
  }
});

function drawPose(view) {
  applyPose(view.bvh, seconds);
  view.bvh.bones.forEach((bone, i) => {
    bone.getWorldPosition(view.positions[i]);
    view.joints.setMatrixAt(i, matrix.makeScale(view.radius, view.radius, view.radius).setPosition(view.positions[i]));
  });
  view.joints.instanceMatrix.needsUpdate = true;
  const attr = view.lines.geometry.getAttribute('position');
  view.segments.forEach(([child, parent], i) => {
    attr.setXYZ(i * 2, ...view.positions[child].toArray());
    attr.setXYZ(i * 2 + 1, ...view.positions[parent].toArray());
  });
  attr.needsUpdate = true;
}

let stageWidth = 0, stageHeight = 0;
function animate(now) {
  requestAnimationFrame(animate);
  const delta = Math.min((now - lastNow) / 1000, .1);
  lastNow = now;
  if (playing && !loading && duration > 0 && !$('summary').open) {
    seconds = (seconds + delta * Number($('speed').value)) % duration;
    updateTransport();
  }
  if (!renderer) return;
  const scroll = $('views-scroll'), bounds = scroll.getBoundingClientRect();
  const width = scroll.clientWidth, height = scroll.clientHeight;
  if (!width || !height) return;
  stage.style.top = `${scroll.offsetTop}px`;
  stage.style.left = `${scroll.offsetLeft}px`;
  stage.style.width = `${width}px`;
  stage.style.height = `${height}px`;
  if (width !== stageWidth || height !== stageHeight) {
    renderer.setSize(width, height);
    // Explicit row height keeps the initial layout at 2x2 even with more clips.
    $('views').style.gridAutoRows = `${Math.max(240, (height - 24) / 2)}px`;
    stageWidth = width; stageHeight = height;
  }
  renderer.setScissorTest(false);
  renderer.clear();
  renderer.setScissorTest(true);
  for (const view of views) {
    if (!view.scene) continue;
    const rect = view.node.getBoundingClientRect();
    const left = rect.left - bounds.left + 1, top = rect.top - bounds.top + 1;
    const viewWidth = rect.width - 2, viewHeight = rect.height - 2;
    const clipLeft = Math.max(0, left), clipTop = Math.max(0, top);
    const clipRight = Math.min(width, left + viewWidth), clipBottom = Math.min(height, top + viewHeight);
    if (clipRight <= clipLeft || clipBottom <= clipTop) continue;
    drawPose(view);
    if (view.reference) drawPose(view.reference);
    view.camera.aspect = viewWidth / Math.max(viewHeight, 1);
    view.camera.updateProjectionMatrix();
    view.controls.update();
    renderer.setViewport(left, height - top - viewHeight, viewWidth, viewHeight);
    renderer.setScissor(clipLeft, height - clipBottom, clipRight - clipLeft, clipBottom - clipTop);
    renderer.render(view.scene, view.camera);
  }
}

function showSummary() {
  const body = $('summary-body'); body.replaceChildren();
  body.append(el('div', `Run: ${report.run_name} | Model: ${report.model_name}`, 'meta'),
    el('div', `Checkpoint: ${report.model_path}`, 'meta'),
    ...(report.task_config ? [el('div', `Task config: ${report.task_config}`, 'meta')] : []),
    el('div', `Generated: ${report.generated}`, 'meta'));
  const values = Object.values(ratings).map(rating => rating.score).sort((a, b) => a - b);
  const cards = el('div', undefined, 'summary-cards');
  const median = values.length ? (values[Math.floor((values.length - 1) / 2)] + values[Math.floor(values.length / 2)]) / 2 : null;
  for (const [label, value] of [
    ['任务总数', report.tasks.length],
    ['人工评分完成', `${values.length} / ${report.tasks.length}`],
    ['人工平均分', values.length ? (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2) : '—'],
    ['人工中位数', median == null ? '—' : median.toFixed(2)],
  ]) { const card = el('div', label, 'summary-card'); card.append(el('strong', String(value))); cards.append(card); }
  body.append(cards, el('h3', 'Overall quality（每个 task 一个分数）'),
    el('p', 'Task quality = 各动作 sqrt(physics × prior) 的中位数；1.0 表示与典型真实动作同等自然。', 'muted'));
  const table = el('table'), head = el('tr');
  const columns = Object.keys(report.summary);
  head.append(el('th', '统计项'), ...columns.map(name => el('th', name))); table.append(head);
  for (const [key, label] of [['mean', 'mean'], ['median', 'median (p50)'], ['p25', 'p25'], ['p75', 'p75'], ['min', 'min'], ['max', 'max']]) {
    const row = el('tr'); row.append(el('td', label), ...columns.map(name => el('td', fmt(report.summary[name][key])))); table.append(row);
  }
  body.append(table, el('h3', '人工评分分布（全报告）'));
  const dist = el('div', undefined, 'distribution');
  const counts = Array.from({ length: 9 }, (_, i) => values.filter(score => score === i + 1).length);
  const max = Math.max(1, ...counts);
  counts.forEach((count, i) => {
    const column = el('div', undefined, 'dist-column');
    const bar = el('div', undefined, 'dist-bar'); bar.style.height = `${count / max * 80}px`;
    column.style.setProperty('--rating-color', color(i + 1));
    column.append(el('span', String(count)), bar, el('span', `${i + 1} 分`, 'dist-label')); dist.append(column);
  });
  body.append(dist, el('p', `未评分 ${report.tasks.length - values.length} 个。评分保存在报告目录的 human_scores.json。`, 'muted'));
  $('summary').showModal();
}
$('summary-open').addEventListener('click', showSummary);
$('summary-close').addEventListener('click', () => $('summary').close());
$('summary').addEventListener('click', event => { if (event.target === $('summary')) {
  const rect = $('summary').getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) $('summary').close();
} });

filterTasks();
updateRatingsPanel();
connect();
requestAnimationFrame(animate);
