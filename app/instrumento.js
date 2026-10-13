const CYCLE = [0.14, 0.36, 0.7, 1, 0.78, 0.4, 0.14, 0.24, 0.44, 0.58, 0.5, 0.36, 0.22, 0.12];
const MARK = [0.3, 0.75, 1, 0.5, 0.2, 0.45, 0.62, 0.36];
const RAMP = [[11, 31, 35], [15, 52, 59], [24, 88, 96], [62, 148, 149], [176, 220, 208]];
const RULER = 18;
const STEPS = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120];

const num = (value, digits = 2) => value.toFixed(digits).replace(".", ",");
const thousands = (value) => String(Math.round(value)).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const bars = (heights) => heights.map((h, n) => `<i style="--h:${h};--n:${n}"></i>`).join("");
const calm = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let stamps = 0;
const stamp = () => `${Date.now()}.${stamps++}`;

const TEMPLATE = `
<header class="top">
  <div class="brand"><span class="mark" aria-hidden="true">${bars(MARK)}</span><span class="word">RespIA</span></div>
  <span class="tag"></span>
</header>
<main class="stage">
  <section class="result" aria-live="polite"><div class="result-in">
    <div class="eyebrow" style="--i:0"><span>Resultado</span><span>patrón acústico</span></div>
    <h1 class="verdict" style="--i:1"></h1>
    <div class="meters" style="--i:2"></div>
    <div class="facts" style="--i:3"></div>
    <p class="note" style="--i:4"></p>
  </div></section>
  <section class="screen">
    <div class="strip"><span class="file"></span><span class="slot"></span></div>
    <div class="scope">
      <canvas></canvas>
      <div class="region"><i class="grip a"></i><i class="grip b"></i></div>
      <i class="playhead"></i><i class="scan"></i>
      <div class="idle"><div class="breath" aria-hidden="true">${bars(CYCLE)}</div></div>
      <p class="alert" role="status"></p>
    </div>
    <div class="strip">
      <div class="transport">
        <button class="play" type="button" aria-label="Reproducir">
          <svg class="go" viewBox="0 0 12 12"><path d="M2.5 1.2v9.6L11 6z"/></svg>
          <svg class="pause" viewBox="0 0 12 12"><path d="M2 1.5h3v9H2zM7 1.5h3v9H7z"/></svg>
        </button>
        <span class="time"></span>
      </div>
      <span class="rate"></span>
    </div>
  </section>
</main>
<footer class="deck">
  <div class="model">
    <span class="label">Modelo</span>
    <div class="switch" role="radiogroup" aria-label="Modelo"><i class="thumb"></i></div>
    <span class="spec"></span>
  </div>
  <button class="analyze" type="button">Analizar<i></i></button>
</footer>
<div class="source"><div class="pads"><span>Ejemplos</span></div><button class="load" type="button">Cargar audio</button></div>
<input type="file" accept=".wav,audio/wav,audio/x-wav" hidden>`;

function create(host) {
  const root = document.createElement("div");
  root.className = "rsp";
  root.dataset.stage = "idle";
  root.innerHTML = TEMPLATE;
  host.appendChild(root);
  const app = { root, $: (selector) => root.querySelector(selector), audio: new Audio(), view: null, span: [0, 0],
    grow: 1, drag: null, seq: null, data: null, api: null };
  app.canvas = app.$("canvas");
  app.source = app.$(".source");
  wire(app);
  return app;
}

function wire(app) {
  const { $ } = app;
  const input = $("input");
  const screen = $(".screen");
  $(".load").addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files[0]) sendFile(app, input.files[0]);
    input.value = "";
  });
  screen.addEventListener("dragover", (event) => { event.preventDefault(); screen.dataset.drop = ""; });
  screen.addEventListener("dragleave", () => delete screen.dataset.drop);
  screen.addEventListener("drop", (event) => {
    event.preventDefault();
    delete screen.dataset.drop;
    if (event.dataTransfer.files[0]) sendFile(app, event.dataTransfer.files[0]);
  });
  $(".play").addEventListener("click", () => (app.root.dataset.playing != null ? stop(app) : play(app)));
  app.audio.addEventListener("ended", () => stop(app));
  $(".analyze").addEventListener("click", () => {
    if (!app.view || app.root.dataset.busy != null) return;
    busy(app, true);
    app.api.setTriggerValue("analyze", { id: app.view.id, t: stamp() });
  });
  $(".scope").addEventListener("dblclick", () => {
    if (app.view) select(app, 0, app.view.duration, true);
  });
  [$(".grip.a"), $(".grip.b")].forEach((grip, index) => {
    grip.addEventListener("pointerdown", (event) => {
      grip.setPointerCapture(event.pointerId);
      app.drag = index;
      stop(app);
    });
    grip.addEventListener("pointermove", (event) => {
      if (app.drag !== index) return;
      const box = app.canvas.getBoundingClientRect();
      const time = clamp((event.clientX - box.left) / box.width, 0, 1) * app.view.duration;
      const gap = Math.min(app.data.min_span, app.view.duration);
      const span = [...app.span];
      span[index] = index === 0 ? Math.min(time, span[1] - gap) : Math.max(time, span[0] + gap);
      select(app, span[0], span[1], false);
    });
    const release = () => {
      if (app.drag !== index) return;
      app.drag = null;
      select(app, app.span[0], app.span[1], true);
    };
    grip.addEventListener("pointerup", release);
    grip.addEventListener("pointercancel", release);
  });
  new ResizeObserver(() => { draw(app); place(app); }).observe($(".scope"));
  document.fonts?.ready.then(() => draw(app));
}

function sendFile(app, file) {
  if (!/\.wav$/i.test(file.name)) return alert(app, "Solo archivos WAV");
  if (file.size > app.data.max_mb * 1048576) return alert(app, `Máximo ${app.data.max_mb} MB`);
  const reader = new FileReader();
  reader.onload = () => {
    busy(app, true);
    app.api.setTriggerValue("file", { name: file.name, data: String(reader.result).split(",")[1], t: stamp() });
  };
  reader.readAsDataURL(file);
}

function alert(app, text) {
  app.$(".alert").textContent = text;
}

function busy(app, on) {
  if (on) app.root.dataset.busy = "";
  else delete app.root.dataset.busy;
}

function select(app, start, end, commit) {
  app.span = [start, end];
  place(app);
  draw(app);
  readout(app);
  if (commit) app.api.setStateValue("selection", { id: app.view.id, start, end });
}

function play(app) {
  if (!app.view) return;
  app.audio.currentTime = app.span[0];
  app.audio.play().then(() => {
    app.root.dataset.playing = "";
    follow(app);
  }).catch(() => {});
}

function stop(app) {
  app.audio.pause();
  delete app.root.dataset.playing;
  readout(app);
}

function follow(app) {
  if (app.root.dataset.playing == null) return;
  if (!app.root.isConnected || app.audio.currentTime >= app.span[1]) return stop(app);
  const width = app.canvas.getBoundingClientRect().width;
  app.$(".playhead").style.transform = `translateX(${(app.audio.currentTime / app.view.duration) * width}px)`;
  readout(app);
  requestAnimationFrame(() => follow(app));
}

function readout(app) {
  if (!app.view) return;
  const [start, end] = app.span;
  const playing = app.root.dataset.playing != null;
  const length = `${num(end - start)} s`;
  app.$(".time").textContent = playing ? `${num(Math.max(0, app.audio.currentTime - start))} / ${length}` : length;
}

function place(app) {
  if (!app.view) return;
  const width = app.canvas.getBoundingClientRect().width;
  const region = app.$(".region");
  region.style.left = `${(app.span[0] / app.view.duration) * width}px`;
  region.style.width = `${((app.span[1] - app.span[0]) / app.view.duration) * width}px`;
}

function paint(spectrogram) {
  const { rows, cols } = spectrogram;
  const cells = Uint8Array.from(atob(spectrogram.cells), (char) => char.charCodeAt(0));
  const canvas = document.createElement("canvas");
  canvas.width = cols;
  canvas.height = rows;
  const context = canvas.getContext("2d");
  const image = context.createImageData(cols, rows);
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      const level = cells[row * cols + col] / 255;
      const at = level * (RAMP.length - 1);
      const low = RAMP[Math.floor(at)];
      const high = RAMP[Math.min(RAMP.length - 1, Math.floor(at) + 1)];
      const mix = at - Math.floor(at);
      const offset = ((rows - 1 - row) * cols + col) * 4;
      for (let channel = 0; channel < 3; channel++) image.data[offset + channel] = low[channel] + (high[channel] - low[channel]) * mix;
      image.data[offset + 3] = 255 * Math.pow(level, 0.7);
    }
  }
  context.putImageData(image, 0, 0);
  return canvas;
}

function open(app, audio) {
  stop(app);
  app.view = { id: audio.id, duration: audio.duration, ticks: audio.spectrogram.ticks,
    peaks: audio.low.map((low, index) => Math.max(Math.abs(low), Math.abs(audio.high[index]))),
    spectrogram: paint(audio.spectrogram) };
  app.audio.src = audio.wav;
  alert(app, "");
  if (calm()) return;
  const began = performance.now();
  const rise = (now) => {
    const progress = clamp((now - began) / 460, 0, 1);
    app.grow = 1 - Math.pow(1 - progress, 4);
    draw(app);
    if (progress < 1 && app.root.isConnected) requestAnimationFrame(rise);
  };
  app.grow = 0;
  requestAnimationFrame(rise);
}

function draw(app) {
  const canvas = app.canvas;
  const box = canvas.getBoundingClientRect();
  const width = Math.round(box.width);
  const height = Math.round(box.height);
  if (!width || !height) return;
  const ratio = window.devicePixelRatio || 1;
  if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
    canvas.width = width * ratio;
    canvas.height = height * ratio;
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const view = app.view;
  if (!view) return;

  const floor = height - RULER;
  const wave = Math.round(floor * 0.56);
  const middle = wave / 2;
  const lane = wave + 14;
  const depth = floor - lane;
  const from = (app.span[0] / view.duration) * width;
  const to = (app.span[1] / view.duration) * width;

  const pitch = 4;
  const count = Math.max(1, Math.floor(width / pitch));
  const bins = view.peaks.length;
  const scale = ((middle - 4) * app.grow) / 127;
  for (let index = 0; index < count; index++) {
    const first = Math.min(bins - 1, Math.floor((index * bins) / count));
    const last = Math.min(bins, Math.max(first + 1, Math.floor(((index + 1) * bins) / count)));
    let peak = 0;
    for (let bin = first; bin < last; bin++) peak = Math.max(peak, view.peaks[bin]);
    const x = index * pitch + 1;
    const half = Math.max(1, peak * scale);
    context.fillStyle = x >= from && x <= to ? "rgba(231, 240, 237, 0.95)" : "rgba(231, 240, 237, 0.26)";
    context.beginPath();
    context.roundRect(x, middle - half, 2, half * 2, 1);
    context.fill();
  }

  context.fillStyle = "rgba(223, 233, 230, 0.035)";
  context.fillRect(0, lane, width, depth);
  context.globalAlpha = app.grow;
  context.drawImage(view.spectrogram, 0, lane, width, depth);
  context.globalAlpha = 1;
  context.fillStyle = "rgba(11, 31, 35, 0.72)";
  context.fillRect(0, lane, from, depth);
  context.fillRect(to, lane, width - to, depth);
  context.font = "9.5px 'IBM Plex Mono', 'Cascadia Mono', Consolas, monospace";
  context.textAlign = "right";
  context.fillStyle = "rgba(223, 233, 230, 0.5)";
  for (const tick of depth < 90 ? view.ticks.filter((item) => item.at === 0 || item.at === 1) : view.ticks) {
    context.fillText(tick.label, width - 5, clamp(lane + depth * (1 - tick.at) + 3, lane + 10, floor - 4));
  }
  context.textAlign = "left";

  const step = STEPS.find((size) => view.duration / size <= Math.max(3, width / 88)) || 300;
  const digits = step < 0.1 ? 2 : step < 1 ? 1 : 0;
  context.font = "10px 'IBM Plex Mono', 'Cascadia Mono', Consolas, monospace";
  for (let time = 0; time <= view.duration + 1e-6; time += step) {
    const x = Math.min(width - 1, (time / view.duration) * width);
    context.fillStyle = "rgba(223, 233, 230, 0.3)";
    context.fillRect(x, floor + 2, 1, 4);
    if (x > width - 30) continue;
    context.fillStyle = "rgba(223, 233, 230, 0.5)";
    context.fillText(num(time, digits), x + 4, height - 2);
  }
}

function pads(app, data) {
  const holder = app.$(".pads");
  let keys = [...holder.querySelectorAll("button")];
  if (keys.length !== data.examples.length) {
    keys.forEach((key) => key.remove());
    keys = data.examples.map((example, index) => {
      const key = document.createElement("button");
      key.type = "button";
      key.className = "pad";
      key.textContent = index + 1;
      key.addEventListener("click", () => {
        busy(app, true);
        app.api.setTriggerValue("example", { index, t: stamp() });
      });
      holder.appendChild(key);
      return key;
    });
  }
  holder.style.display = keys.length ? "" : "none";
  keys.forEach((key, index) => {
    key.title = data.examples[index].title;
    key.setAttribute("aria-label", `Ejemplo ${index + 1}`);
    key.setAttribute("aria-pressed", String(data.example === index));
  });
}

function models(app, data) {
  const holder = app.$(".switch");
  let keys = [...holder.querySelectorAll("button")];
  if (keys.length !== data.models.length) {
    keys.forEach((key) => key.remove());
    keys = data.models.map((model) => {
      const key = document.createElement("button");
      key.type = "button";
      key.setAttribute("role", "radio");
      key.textContent = model.name;
      key.addEventListener("click", () => {
        if (app.data.model === model.key) return;
        app.data.model = model.key;
        models(app, app.data);
        app.api.setStateValue("model", model.key);
      });
      holder.appendChild(key);
      return key;
    });
    app.$(".thumb").style.width = `calc(${100 / Math.max(1, keys.length)}% - ${keys.length > 1 ? 3 : 6}px)`;
  }
  const active = Math.max(0, data.models.findIndex((model) => model.key === data.model));
  keys.forEach((key, index) => key.setAttribute("aria-checked", String(index === active)));
  holder.dataset.index = active;
  app.$(".spec").textContent = data.models[active]?.spec ?? "";
}

function outcome(app, data) {
  if (data.stage !== "result") return;
  const result = data.result;
  const verdict = app.$(".verdict");
  verdict.classList.toggle("none", !result);
  verdict.textContent = result ? result.name : "Sin resultado";

  const meters = app.$(".meters");
  const rows = result ? result.probabilities : [];
  if (meters.dataset.keys !== rows.map((row) => row.key).join()) {
    meters.dataset.keys = rows.map((row) => row.key).join();
    meters.replaceChildren();
    rows.forEach((row) => {
      const name = document.createElement("span");
      name.className = "n";
      name.textContent = row.name;
      const meter = document.createElement("div");
      meter.className = "meter";
      meter.appendChild(document.createElement("i"));
      const value = document.createElement("span");
      value.className = "v";
      meters.append(name, meter, value);
    });
  }
  rows.forEach((row, index) => {
    const [name, meter, value] = [0, 1, 2].map((offset) => meters.children[index * 3 + offset]);
    const on = row.key === result.label;
    [name, meter, value].forEach((node) => node.classList.toggle("on", on));
    meter.firstChild.style.setProperty("--v", row.value);
    value.textContent = `${num(row.value * 100, 1)} %`;
  });

  const facts = app.$(".facts");
  facts.replaceChildren();
  const fact = (text, tone) => {
    const item = document.createElement("span");
    if (tone) item.className = tone;
    item.textContent = text;
    facts.appendChild(item);
  };
  if (result?.votes) fact(`${result.votes.agree} de ${result.votes.total} votos`);
  if (result?.annotation) fact(`anotación: ${result.annotation.name} ${result.annotation.match ? "✓" : "✕"}`, result.annotation.match ? "ok" : "miss");

  const notes = [...data.messages];
  if (result?.doubt) notes.unshift(`Caso dudoso: la probabilidad más alta es la de ${result.doubt}.`);
  app.$(".note").textContent = notes.join(" ");
}

function update(app, data) {
  const { root, $ } = app;
  app.data = data;
  if (data.seq !== app.seq) {
    app.seq = data.seq;
    busy(app, false);
  }
  root.dataset.stage = data.stage;
  $(".tag").textContent = data.tag;
  (data.stage === "idle" ? $(".idle") : $(".slot")).appendChild(app.source);
  pads(app, data);
  models(app, data);

  if (!data.audio) {
    stop(app);
    app.view = null;
    app.audio.removeAttribute("src");
  } else {
    if (!app.view || app.view.id !== data.audio.id) open(app, data.audio);
    $(".file").textContent = data.audio.name;
    $(".rate").textContent = `${thousands(data.audio.sample_rate)} Hz`;
    if (app.drag == null) app.span = [...data.selection];
    root.toggleAttribute("data-trim", data.audio.duration > 2 * data.min_span);
  }
  if (data.error) alert(app, data.error);
  outcome(app, data);
  place(app);
  draw(app);
  readout(app);
}

export default function (component) {
  const host = component.parentElement;
  let app = host.__respia;
  if (!app || !app.root.isConnected) app = host.__respia = create(host);
  app.api = component;
  update(app, component.data);
  return () => setTimeout(() => { if (!app.root.isConnected) app.audio.pause(); }, 0);
}
