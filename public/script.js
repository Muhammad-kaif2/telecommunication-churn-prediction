(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const form = $("form"), go = $("go");
  const states = { idle: $("sIdle"), load: $("sLoad"), res: $("sRes"), err: $("sErr") };
  const REDUCE = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let examples = {};

  const show = (name) => Object.entries(states).forEach(([k, el]) => (el.hidden = k !== name));

  // ---- API metadata: states, ranges, examples, model note ---------------------------------
  async function loadMeta() {
    const pill = $("apiPill");
    try {
      const res = await fetch("/api/meta");
      if (!res.ok) throw new Error(res.status);
      const m = await res.json();
      const sel = $("state");
      m.states.forEach((s) => sel.add(new Option(s, s)));
      examples = m.examples || {};
      Object.entries(m.ranges).forEach(([col, [lo, hi]]) => {
        const el = document.querySelector(`[data-hint="${col}"]`);
        if (el) el.textContent = `Seen in data: ${lo}–${hi}`;
      });
      const t = m.test_metrics, n = t.tp + t.fn;
      $("modelNote").textContent =
        `Tuned LightGBM trained on 3,333 telecom customers. On held-out data it caught ${t.tp} of ${n} churners ` +
        `(${Math.round((t.tp / n) * 100)}% recall) with ${Math.round((t.tp / (t.tp + t.fp)) * 100)}% precision. ` +
        `Overall churn rate in the data: ${Math.round(m.base_rate * 100)}%.`;
      pill.className = "api-pill on"; $("apiText").textContent = "API connected";
    } catch {
      pill.className = "api-pill off"; $("apiText").textContent = "API unreachable";
    }
  }

  // ---- form helpers ------------------------------------------------------------------------
  const wrapOf = (name) => {
    const el = form.elements[name];
    const node = el instanceof RadioNodeList ? el[0] : el;
    return node && node.closest(".f");
  };
  const setErr = (name, msg) => {
    const w = wrapOf(name); if (!w) return;
    w.classList.add("bad");
    const e = w.querySelector(`[data-err="${name}"]`); if (e) e.textContent = msg;
  };
  const clearErrs = () => form.querySelectorAll(".f.bad").forEach((w) => {
    w.classList.remove("bad"); w.querySelectorAll(".err").forEach((e) => (e.textContent = ""));
  });
  form.addEventListener("input", (e) => {
    const w = e.target.closest(".f"); if (w) { w.classList.remove("bad"); w.querySelectorAll(".err").forEach((x) => (x.textContent = "")); }
  });

  const INTS = ["account_length", "total_day_calls", "total_eve_calls", "total_night_calls", "total_intl_calls", "customer_service_calls"];
  const FLOATS = ["total_day_charge", "total_eve_charge", "total_night_charge", "total_intl_charge"];

  function collect() {
    const fd = new FormData(form), p = {};
    INTS.forEach((k) => (p[k] = fd.get(k) === "" ? NaN : parseInt(fd.get(k), 10)));
    FLOATS.forEach((k) => (p[k] = fd.get(k) === "" ? NaN : parseFloat(fd.get(k))));
    p.state = fd.get("state") || "";
    p.area_code = fd.get("area_code") ? parseInt(fd.get("area_code"), 10) : null;
    p.international_plan = fd.get("international_plan") || "";
    p.voice_mail_plan = fd.get("voice_mail_plan") || "";
    return p;
  }

  function validate(p) {
    let first = null;
    const bad = (k, m) => { setErr(k, m); first = first || k; };
    ["state", "area_code", "international_plan", "voice_mail_plan"].forEach((k) => { if (!p[k]) bad(k, "Required."); });
    [...INTS, ...FLOATS].forEach((k) => {
      const el = form.elements[k];
      if (Number.isNaN(p[k])) bad(k, "Required.");
      else if (p[k] < +el.min || p[k] > +el.max) bad(k, `Must be ${el.min}–${el.max}.`);
    });
    if (first) { const el = form.elements[first]; (el instanceof RadioNodeList ? el[0] : el).focus(); }
    return !first;
  }

  // ---- examples / clear --------------------------------------------------------------------
  function fill(ex) {
    clearErrs();
    Object.entries(ex).forEach(([col, val]) => {
      const name = col.replace(/ /g, "_"), el = form.elements[name]; if (!el) return;
      if (el instanceof RadioNodeList) el.value = String(val); else el.value = String(val);
    });
  }
  document.querySelectorAll("[data-example]").forEach((b) =>
    b.addEventListener("click", () => { const ex = examples[b.dataset.example]; if (ex) fill(ex); }));
  $("clear").addEventListener("click", () => { form.reset(); clearErrs(); show("idle"); });

  // ---- result ------------------------------------------------------------------------------
  function fmtPct(p) { const v = p * 100; return v < 1 ? "<1" : v > 99 ? ">99" : String(Math.round(v)); }

  function render(r) {
    const p = r.churn_probability, shown = fmtPct(p);
    $("meter").dataset.level = r.risk_level;
    const v = $("verdict"); v.className = "verdict " + r.risk_level;
    v.textContent = r.risk_level === "High" ? "High risk · likely to churn"
                  : r.risk_level === "Elevated" ? "Elevated risk · watch closely" : "Low risk · likely to stay";
    $("sub").textContent = r.will_churn
      ? "The model predicts this customer will leave. A proactive retention offer is worth considering."
      : r.risk_level === "Elevated"
        ? "Below the 50% decision threshold, but above the quarter mark. Worth keeping an eye on."
        : "The model predicts this customer will stay.";
    const box = $("insights"); box.replaceChildren();
    r.insights.forEach((i) => { const d = document.createElement("div"); d.className = "ins " + i.tone; d.textContent = i.text; box.appendChild(d); });

    show("res");
    const fill = $("fill"), num = $("pct");
    fill.style.transition = "none"; fill.style.width = "0%";
    requestAnimationFrame(() => requestAnimationFrame(() => {
      fill.style.transition = ""; fill.style.width = Math.max(p * 100, 1.5) + "%";
    }));
    if (REDUCE || shown.startsWith("<") || shown.startsWith(">")) { num.textContent = shown; return; }
    const target = +shown, t0 = performance.now();
    (function tick(now) {
      const t = Math.min(1, (now - t0) / 900);
      num.textContent = Math.round(target * (1 - Math.pow(1 - t, 3)));
      if (t < 1) requestAnimationFrame(tick);
    })(t0);
  }

  function fail(title, text) { $("errTitle").textContent = title; $("errText").textContent = text; show("err"); }

  form.addEventListener("submit", async (e) => {
    e.preventDefault(); clearErrs();
    const payload = collect();
    if (!validate(payload)) return;
    go.disabled = true; go.classList.add("loading"); show("load");
    try {
      const res = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      if (res.status === 422) {
        const body = await res.json().catch(() => null);
        (body?.detail || []).forEach((d) => { const k = d.loc?.[d.loc.length - 1]; if (k) setErr(k, d.msg.replace(/^Value error, /, "")); });
        fail("Check the highlighted fields", "The API rejected some values. Correct them and try again.");
        return;
      }
      if (!res.ok) { fail("Prediction failed", `The server responded with status ${res.status}.`); return; }
      render(await res.json());
    } catch {
      fail("Can't reach the server", "Check your connection, and that the backend is running, then try again.");
    } finally { go.disabled = false; go.classList.remove("loading"); }
  });

  $("again").addEventListener("click", () => { show("idle"); form.scrollIntoView({ behavior: REDUCE ? "auto" : "smooth", block: "start" }); });
  $("errBack").addEventListener("click", () => show("idle"));

  loadMeta();
})();
