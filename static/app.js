(function () {
  "use strict";

  function $(id) { return document.getElementById(id); }
  function on(el, ev, fn) { el.addEventListener(ev, fn); }

  async function api(path, body) {
    const res = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || ("Request failed (" + res.status + ")"));
    return data;
  }

  function showBanner(el, msg, kind) {
    el.textContent = msg;
    el.className = "banner " + kind;
    el.hidden = false;
  }
  function hideBanner(el) { el.hidden = true; }

  function setChip(el, text, on_) {
    el.textContent = text;
    el.classList.toggle("on", !!on_);
  }

  /* ============================================================ STEP 1 */
  const connectBtn = $("connectBtn");
  on(connectBtn, "click", async () => {
    const apiKey = $("apiKey").value.trim();
    const model = $("modelName").value.trim() || "openai/gpt-5.4-nano";
    if (!apiKey) { showBanner($("connectBanner"), "Paste an API key first.", "error"); return; }
    connectBtn.disabled = true;
    connectBtn.textContent = "Connecting…";
    hideBanner($("connectBanner"));
    try {
      const r = await api("/api/configure", { api_key: apiKey, model: model });
      showBanner($("connectBanner"), "Connected — a real test call to " + r.model + " succeeded.", "ok");
      setChip($("chipKey"), "🔑 key: connected", true);
    } catch (e) {
      showBanner($("connectBanner"), e.message, "error");
      setChip($("chipKey"), "🔑 key: not set", false);
    } finally {
      connectBtn.disabled = false;
      connectBtn.textContent = "Connect";
    }
  });

  /* ============================================================ STEP 2 */
  let dataSource = "builtin";
  let dataFormat = "csv";
  let moduleChoice = "cot";

  document.querySelectorAll('#moduleSeg [data-module]').forEach((btn) => {
    on(btn, "click", () => {
      moduleChoice = btn.dataset.module;
      document.querySelectorAll('#moduleSeg [data-module]').forEach((b) => b.setAttribute("aria-selected", b === btn ? "true" : "false"));
    });
  });

  document.querySelectorAll('[data-source]').forEach((btn) => {
    on(btn, "click", () => {
      dataSource = btn.dataset.source;
      document.querySelectorAll('[data-source]').forEach((b) => b.setAttribute("aria-selected", b === btn ? "true" : "false"));
      $("builtinPanel").hidden = dataSource !== "builtin";
      $("customPanel").hidden = dataSource !== "custom";
    });
  });
  document.querySelectorAll('[data-format]').forEach((btn) => {
    on(btn, "click", () => {
      dataFormat = btn.dataset.format;
      document.querySelectorAll('[data-format]').forEach((b) => b.setAttribute("aria-selected", b === btn ? "true" : "false"));
    });
  });
  on($("fileInput"), "change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (file.name.toLowerCase().endsWith(".json")) {
      dataFormat = "json";
      document.querySelectorAll('[data-format]').forEach((b) => b.setAttribute("aria-selected", b.dataset.format === "json" ? "true" : "false"));
    }
    const reader = new FileReader();
    reader.onload = () => { $("rawData").value = reader.result; };
    reader.readAsText(file);
  });

  const loadDatasetBtn = $("loadDatasetBtn");
  on(loadDatasetBtn, "click", async () => {
    loadDatasetBtn.disabled = true;
    loadDatasetBtn.textContent = "Loading…";
    hideBanner($("datasetBanner"));
    try {
      const payload = dataSource === "builtin"
        ? { source: "builtin", module: moduleChoice }
        : {
            source: "custom",
            module: moduleChoice,
            format: dataFormat,
            raw_text: $("rawData").value,
            instructions: $("customInstructions").value,
          };
      const r = await api("/api/dataset", payload);
      showBanner($("datasetBanner"), "Dataset loaded.", "ok");
      setChip($("chipData"), "📄 data: " + r.n_train + "/" + r.n_dev + " train/dev", true);
      setChip($("chipOpt"), "⚙️ optimized: no", false);
      $("optimizedTab").disabled = true;
      $("optimizedTab").setAttribute("aria-selected", "false");
      $("whichProgram").querySelector('[data-which="baseline"]').setAttribute("aria-selected", "true");
      $("statTrain").textContent = r.n_train;
      $("statDev").textContent = r.n_dev;
      $("statCats").textContent = r.categories.length;
      $("statUrg").textContent = r.urgencies.length;
      const table = $("samplePreview");
      table.innerHTML =
        "<tr><th>ticket</th><th>category</th><th>urgency</th></tr>" +
        r.sample.map((s) => "<tr><td>" + escapeHtml(s.ticket) + "</td><td>" + escapeHtml(s.category) + "</td><td>" + escapeHtml(s.urgency) + "</td></tr>").join("");
      $("datasetSummary").hidden = false;
      renderTrace($("datasetTrace"), r.trace);
      $("promptView").hidden = true;
      $("classifyLive").hidden = true;
      $("resultView").hidden = true;
      $("winnerBlock").hidden = true;
      $("optLive").hidden = true;
    } catch (e) {
      showBanner($("datasetBanner"), e.message, "error");
    } finally {
      loadDatasetBtn.disabled = false;
      loadDatasetBtn.textContent = "Load dataset";
    }
  });

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function fmtMs(ms) {
    if (ms < 1) return "<1ms";
    if (ms < 1000) return Math.round(ms) + "ms";
    return (ms / 1000).toFixed(2) + "s";
  }

  // Renders a full trace (returned all at once from a fast, non-streamed call)
  // with a staggered fade-in purely for readability - the ms shown per row is real.
  function renderTrace(container, events) {
    container.innerHTML = (events || [])
      .map((e, i) => (
        '<div class="trace-row" style="animation-delay:' + (i * 70) + 'ms"><span class="dt">+' +
        fmtMs(e.ms) + '</span><span class="msg">' + escapeHtml(e.msg) + "</span></div>"
      ))
      .join("");
  }

  // Appends one row live as a Server-Sent Event arrives - genuinely real-time.
  function appendTraceRow(container, evt, waiting) {
    const row = document.createElement("div");
    row.className = "trace-row" + (waiting ? " waiting" : "");
    row.innerHTML = '<span class="dt">+' + fmtMs(evt.ms) + '</span><span class="msg">' + escapeHtml(evt.msg) + "</span>";
    const prevWaiting = container.querySelector(".trace-row.waiting");
    if (prevWaiting) prevWaiting.classList.remove("waiting");
    container.appendChild(row);
    container.scrollTop = container.scrollHeight;
  }

  /* ============================================================ STEP 3 */
  let which = "baseline";
  document.querySelectorAll('#whichProgram [data-which]').forEach((btn) => {
    on(btn, "click", () => {
      if (btn.disabled) return;
      which = btn.dataset.which;
      document.querySelectorAll('#whichProgram [data-which]').forEach((b) => b.setAttribute("aria-selected", b === btn ? "true" : "false"));
    });
  });

  on($("previewBtn"), "click", async () => {
    hideBanner($("tryBanner"));
    try {
      const r = await api("/api/prompt-preview", { ticket: $("tryTicket").value, which: which });
      $("fdText").textContent = r.breakdown.field_description;
      $("fsText").textContent = r.breakdown.field_structure;
      $("tdText").textContent = r.breakdown.task_description;
      $("demoNote").textContent = r.n_demos === 0
        ? "0 few-shot demos in this prompt right now — " + r.messages.length + " messages total."
        : r.n_demos + " few-shot demo(s) baked in from optimization — " + r.messages.length + " messages total.";
      renderTrace($("previewTrace"), r.trace);
      $("promptView").hidden = false;
    } catch (e) {
      showBanner($("tryBanner"), e.message, "error");
    }
  });

  const classifyBtn = $("classifyBtn");
  on(classifyBtn, "click", async () => {
    hideBanner($("tryBanner"));
    classifyBtn.disabled = true;
    classifyBtn.textContent = "Calling the LM…";
    $("classifyTrace").innerHTML = "";
    $("classifyLive").hidden = false;
    $("resultView").hidden = true;
    const t0 = performance.now();

    try {
      const res = await fetch("/api/classify/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticket: $("tryTicket").value, which: which }),
      });
      if (!res.ok || !res.body) throw new Error("Request failed (" + res.status + ")");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let done = null;
      let errMsg = null;

      while (true) {
        const { value, done: streamDone } = await reader.read();
        if (streamDone) break;
        buf += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n\n")) !== -1) {
          const chunk = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          if (!chunk.startsWith("data: ")) continue;
          const evt = JSON.parse(chunk.slice(6));
          if (evt.type === "step") appendTraceRow($("classifyTrace"), evt, false);
          else if (evt.type === "waiting") appendTraceRow($("classifyTrace"), evt, true);
          else if (evt.type === "prompt") {
            $("fdText").textContent = evt.breakdown.field_description;
            $("fsText").textContent = evt.breakdown.field_structure;
            $("tdText").textContent = evt.breakdown.task_description;
            $("demoNote").textContent = evt.n_demos === 0
              ? "0 few-shot demos in this prompt right now — " + evt.n_messages + " messages total."
              : evt.n_demos + " few-shot demo(s) baked in from optimization — " + evt.n_messages + " messages total.";
            $("previewTrace").innerHTML = "";
            $("promptView").hidden = false;
          }
          else if (evt.type === "error") errMsg = evt.msg;
          else if (evt.type === "done") done = evt;
        }
      }

      if (errMsg) throw new Error(errMsg);
      if (!done) throw new Error("The stream ended without a result.");

      $("resCategory").textContent = done.category;
      $("resUrgency").textContent = done.urgency;
      $("resLatency").textContent = Math.round(performance.now() - t0) + " ms (wall clock)";
      if (done.reasoning) {
        $("resReasoning").textContent = done.reasoning;
        $("resReasoningWrap").hidden = false;
      } else {
        $("resReasoningWrap").hidden = true;
      }
      $("resultView").hidden = false;
    } catch (e) {
      showBanner($("tryBanner"), e.message, "error");
    } finally {
      classifyBtn.disabled = false;
      classifyBtn.textContent = "Classify (real call)";
    }
  });

  /* ============================================================ STEP 4 */
  const optimizeBtn = $("optimizeBtn");
  let pollTimer = null;

  on(optimizeBtn, "click", async () => {
    hideBanner($("optimizeBanner"));
    optimizeBtn.disabled = true;
    optimizeBtn.textContent = "Running…";
    $("optLive").hidden = false;
    $("optSummary").hidden = true;
    $("chartWrap").hidden = true;
    $("winnerBlock").hidden = true;
    $("optLog").textContent = "";
    try {
      await api("/api/optimize/start", {
        metric: $("metricSelect").value,
        optimizer: $("optimizerSelect").value,
      });
      pollTimer = setInterval(pollOptimize, 1000);
    } catch (e) {
      showBanner($("optimizeBanner"), e.message, "error");
      optimizeBtn.disabled = false;
      optimizeBtn.textContent = "Run optimization";
    }
  });

  async function pollOptimize() {
    let r;
    try { r = await api("/api/optimize/status"); } catch (e) { return; }

    $("optLog").textContent = r.log.join("\n");
    $("optLog").scrollTop = $("optLog").scrollHeight;

    if (r.trials && r.trials.length) {
      $("chartWrap").hidden = false;
      renderChart(r.trials);
    }

    if (r.status === "done") {
      clearInterval(pollTimer);
      optimizeBtn.disabled = false;
      optimizeBtn.textContent = "Run optimization";
      $("optSummary").hidden = false;
      $("optBefore").textContent = r.before.toFixed(1) + "%";
      $("optAfter").textContent = r.after.toFixed(1) + "%";
      $("optDelta").textContent = (r.after - r.before >= 0 ? "+" : "") + (r.after - r.before).toFixed(1);
      $("winnerText").textContent = r.instructions;
      $("winnerDemos").textContent = r.n_demos + " few-shot demo(s) selected.";
      $("winnerBlock").hidden = false;
      setChip($("chipOpt"), "⚙️ optimized: yes (" + r.after.toFixed(1) + "%)", true);
      $("optimizedTab").disabled = false;
      showBanner($("optimizeBanner"), "Done — try it on your own text in step 3, switched to \"Optimized\".", "ok");
    } else if (r.status === "error") {
      clearInterval(pollTimer);
      optimizeBtn.disabled = false;
      optimizeBtn.textContent = "Run optimization";
      showBanner($("optimizeBanner"), r.error || "Optimization failed.", "error");
    }
  }

  function renderChart(trials) {
    const svg = $("trialChart");
    const W = 640, H = 220, ML = 38, MR = 12, MT = 14, MB = 26;
    const plotW = W - ML - MR;
    const n = Math.max(trials.length, 2);
    const scores = trials.map((t) => t.score);
    const yMin = Math.max(0, Math.floor((Math.min(...scores) - 5) / 10) * 10);
    const yMax = Math.min(100, Math.ceil((Math.max(...scores) + 5) / 10) * 10);
    const xAt = (i) => ML + (i * plotW) / (n - 1);
    const yAt = (v) => MT + ((yMax - v) / (yMax - yMin)) * (H - MT - MB);

    let best = -Infinity;
    const bestPts = [];
    trials.forEach((t) => { best = Math.max(best, t.score); bestPts.push(best); });

    const parts = [];
    const steps = 4;
    for (let s = 0; s <= steps; s++) {
      const v = yMin + ((yMax - yMin) * s) / steps;
      const y = yAt(v);
      parts.push('<line class="gridline" x1="' + ML + '" x2="' + (W - MR) + '" y1="' + y + '" y2="' + y + '"></line>');
      parts.push('<text class="axislabel" x="' + (ML - 6) + '" y="' + (y + 3) + '" text-anchor="end">' + Math.round(v) + "%</text>");
    }
    trials.forEach((t, i) => {
      parts.push('<text class="axislabel" x="' + xAt(i) + '" y="' + (H - 6) + '" text-anchor="middle">' + t.n + "</text>");
    });

    if (bestPts.length > 1) {
      let d = "M " + xAt(0) + " " + yAt(bestPts[0]);
      for (let i = 1; i < bestPts.length; i++) {
        d += " L " + xAt(i) + " " + yAt(bestPts[i - 1]) + " L " + xAt(i) + " " + yAt(bestPts[i]);
      }
      parts.push('<path d="' + d + '" fill="none" stroke="var(--accent2)" stroke-width="2"></path>');
    }
    if (trials.length > 1) {
      let d2 = "M " + trials.map((t, i) => xAt(i) + " " + yAt(t.score)).join(" L ");
      parts.push('<path d="' + d2 + '" fill="none" stroke="var(--accent)" stroke-width="1.5" opacity="0.55"></path>');
    }
    trials.forEach((t, i) => {
      const isBest = t.score === best;
      parts.push('<circle cx="' + xAt(i) + '" cy="' + yAt(t.score) + '" r="' + (isBest ? 5 : 4) + '" fill="' + (isBest ? "var(--accent2)" : "var(--accent)") + '" stroke="var(--surface)" stroke-width="1.5"><title>Trial ' + t.n + " — " + t.score.toFixed(2) + "% — " + escapeHtml(t.label) + "</title></circle>");
    });

    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.innerHTML = parts.join("");
  }

  /* ============================================================ boot */
  (async function boot() {
    try {
      const s = await api("/api/status");
      if (s.auto_configured) {
        $("connectForm").hidden = true;
        $("connectHint").hidden = true;
        $("step1Title").textContent = "Connected";
        $("step1Guide").textContent = "This copy of the demo is already connected to a model (" + s.model + ") by whoever's hosting it — you don't need to provide your own key. Skip straight to step 2.";
        showBanner($("autoConfigBanner"), "Already connected to " + s.model + ".", "ok");
        setChip($("chipKey"), "🔑 key: connected", true);
      } else if (s.configured) {
        setChip($("chipKey"), "🔑 key: connected", true);
      }
      if (s.has_dataset) setChip($("chipData"), "📄 data: " + s.n_train + "/" + s.n_dev + " train/dev", true);
      if (s.has_optimized) { setChip($("chipOpt"), "⚙️ optimized: yes", true); $("optimizedTab").disabled = false; }
    } catch (e) { /* server just started, ignore */ }
  })();
})();
