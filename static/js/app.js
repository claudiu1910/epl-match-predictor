/* EPL Predictor dashboard: API calls, fixture cards, deep-dive charts, simulator, sync. */
(() => {
  "use strict";

  const COLORS = { home: "#38bdf8", draw: "#94a3b8", away: "#fbbf24", grid: "rgba(148,163,184,0.12)", text: "#94a3b8" };
  const STYLE_KEY = { "High-Press Possession": "possession", "Counter / Low Block": "low_block", "Direct Transition": "transition" };
  const S = { days: "", ready: false, charts: {}, teams: [], allTeams: null, status: null, syncing: false };

  // ------------------------------------------------------------------ helpers
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (p, d = 0) => (p == null || Number.isNaN(p) ? "—" : `${(p * 100).toFixed(d)}%`);
  const num = (x, d = 2) => (x == null || Number.isNaN(x) ? "—" : Number(x).toFixed(d));
  const signed = (x, d = 1) => (x == null ? "—" : `${x > 0 ? "+" : ""}${Number(x).toFixed(d)}`);
  const kickoff = (iso, withDay = true) => {
    if (!iso) return "Date TBC";
    const opts = withDay ? { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }
                         : { hour: "2-digit", minute: "2-digit" };
    return new Intl.DateTimeFormat(undefined, opts).format(new Date(iso));
  };
  const dateOnly = (iso) => (iso ? new Intl.DateTimeFormat(undefined, { weekday: "short", day: "numeric", month: "short" }).format(new Date(iso)) : "—");
  const relTime = (iso) => {
    if (!iso) return "never";
    const mins = Math.round((Date.now() - new Date(iso).getTime()) / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins} min ago`;
    const h = Math.round(mins / 60);
    return h < 48 ? `${h} h ago` : `${Math.round(h / 24)} days ago`;
  };
  const styleChip = (style) => style ? `<span class="style-chip style-${STYLE_KEY[style] || "transition"}" title="Tactical archetype (last 10 matches)">${esc(style)}</span>` : "";

  async function api(path, opts = {}) {
    const headers = { "Content-Type": "application/json", ...(opts.headers || {}) };
    const res = await fetch(path, { ...opts, headers });
    let body = null;
    try { body = await res.json(); } catch (_) { /* empty body */ }
    if (!res.ok) {
      const err = new Error((body && body.detail) || res.statusText || `HTTP ${res.status}`);
      err.status = res.status;
      err.body = body;
      throw err;
    }
    return body;
  }

  let toastTimer;
  function toast(message, kind = "info", ms = 5000) {
    const el = $("#toast");
    el.className = `fixed bottom-5 right-5 z-[60] max-w-sm rounded-xl px-4 py-3 text-sm shadow-xl ring-1 ${kind}`;
    el.textContent = message;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.add("hidden"), ms);
  }

  function chart(key, canvas, config) {
    if (S.charts[key]) S.charts[key].destroy();
    S.charts[key] = new Chart(canvas, config);
    return S.charts[key];
  }

  Chart.defaults.color = COLORS.text;
  Chart.defaults.font.family = "Inter, ui-sans-serif, system-ui, sans-serif";
  Chart.defaults.font.size = 11;
  Chart.defaults.borderColor = COLORS.grid;
  Chart.defaults.animation.duration = 450;

  // ------------------------------------------------------------------ status
  async function pollStatus() {
    try {
      const st = await api("/api/status");
      S.status = st;
      renderStatus(st);
      if (st.ready && !S.ready) {
        S.ready = true;
        loadAll();
      }
      if (!st.ready) setTimeout(pollStatus, 2500);
    } catch (err) {
      renderPill("error", "Server unreachable");
      setTimeout(pollStatus, 5000);
    }
  }

  function renderPill(kind, text) {
    const colours = { ready: "bg-emerald-400", loading: "bg-amber-400 animate-pulse", error: "bg-rose-500", syncing: "bg-sky-400 animate-pulse" };
    $("#status-pill").innerHTML = `<span class="dot ${colours[kind]}"></span><span>${esc(text)}</span>`;
  }

  function renderStatus(st) {
    if (S.syncing || (st.sync_job && st.sync_job.state === "running")) renderPill("syncing", "Syncing…");
    else if (st.ready) renderPill("ready", "Live");
    else if (st.error) renderPill("error", "Load failed");
    else renderPill("loading", "Training model…");
    if (st.error && !st.ready) {
      $("#gw-subtitle").textContent = `The model could not load: ${st.error}. Try “Sync data”.`;
    }
    $("#season-label").textContent = st.season ? `Premier League ${st.season}` : "Premier League";
    $("#stat-latest").textContent = st.latest_result ? `${dateOnly(st.latest_result)} · ${st.current_season_matches} played` : "—";
    $("#stat-train").textContent = st.model ? `${st.model.n_train.toLocaleString()} · through ${dateOnly(st.model.trained_through)}` : "—";
    $("#stat-sync").textContent = relTime(st.last_sync);
    $("#stat-next").textContent = st.scheduler && st.scheduler.next_run ? kickoff(st.scheduler.next_run) : (st.scheduler && st.scheduler.enabled === false ? "Disabled" : "—");
  }

  function loadAll() {
    loadFixtures();
    loadTeams().then(loadModel);  // the tactics table lists clubs per style
  }

  // ------------------------------------------------------------------ fixtures
  async function loadFixtures() {
    const grid = $("#fixture-grid");
    grid.innerHTML = '<div class="skeleton h-56"></div>'.repeat(3);
    try {
      const data = await api(`/api/fixtures${S.days ? `?days=${S.days}` : ""}`);
      renderFixtures(data);
    } catch (err) {
      grid.innerHTML = `<div class="panel p-6 text-sm text-rose-300 sm:col-span-2 xl:col-span-3">Could not load fixtures: ${esc(err.message)}</div>`;
    }
  }

  function renderFixtures(data) {
    const fx = data.fixtures || [];
    $("#gw-title").textContent = `${data.label}${data.season ? ` · ${data.season}` : ""}`;
    if (fx.length) {
      const first = fx[0].kickoff_utc, last = fx[fx.length - 1].kickoff_utc;
      const value = data.value_bets ? ` · ${data.value_bets} value flag${data.value_bets > 1 ? "s" : ""}` : "";
      const priced = fx.some((f) => f.has_odds);
      $("#gw-subtitle").textContent = `${fx.length} fixtures · ${dateOnly(first)} – ${dateOnly(last)}${value}${priced ? "" : " · bookmaker prices not published yet"}`;
    } else {
      $("#gw-subtitle").textContent = "No upcoming fixtures found. Try syncing.";
    }
    const pending = data.pending || [];
    const note = $("#pending-note");
    if (pending.length) {
      note.textContent = `${pending.length} earlier fixture(s) kicked off without a result yet (in play, awaiting sync or postponed).`;
      note.classList.remove("hidden");
    } else note.classList.add("hidden");
    $("#fixture-grid").innerHTML = fx.map(cardHtml).join("") ||
      '<div class="panel p-6 text-sm text-slate-400 sm:col-span-2 xl:col-span-3">Nothing scheduled in this window.</div>';
    $$(".fixture-card").forEach((card) => card.addEventListener("click", () => openFixture(card.dataset.id)));
  }

  function cardHtml(f) {
    const p = f.probs;
    const best = f.pick.outcome;
    const value = (f.value || []).slice().sort((a, b) => b.ev - a.ev);
    const evBadge = value.length
      ? `<span class="ev-badge ${value[0].longshot ? "caution" : ""}" title="${esc(value.map((v) => `${v.label} @ ${v.odds} (EV ${(v.ev * 100).toFixed(1)}%)`).join(" · "))}">+EV ${(value[0].ev * 100).toFixed(0)}%</span>` : "";
    const gw = f.gameweek ? `<span class="chip">GW${f.gameweek}</span>` : "";
    const row = (side, name) => `
      <div class="flex items-center gap-2 min-w-0">
        <span class="h-2.5 w-2.5 rounded-full shrink-0" style="background:${COLORS[side]}"></span>
        <span class="truncate font-semibold ${best === side ? "text-slate-50" : "text-slate-300"}">${esc(name)}</span>
        ${styleChip(f.styles[side])}
        <span class="ml-auto tabular-nums text-sm ${best === side ? "font-semibold text-slate-50" : "text-slate-400"}">${pct(p[side])}</span>
      </div>`;
    const warn = ["home", "away"].filter((s) => (f.availability || {})[s] >= 0.05)
      .map((s) => `<span class="chip warn-chip" title="Key attackers missing (FPL availability)">⚠ ${esc(s === "home" ? f.home_short : f.away_short)} attack −${Math.round(f.availability[s] * 100)}%</span>`).join("");
    const bo = f.most_likely;
    return `
      <button type="button" class="fixture-card" data-id="${esc(f.id)}" aria-label="${esc(`${f.home} v ${f.away}, details`)}">
        <div class="flex items-center justify-between gap-2 text-xs text-slate-400">
          <span>${esc(kickoff(f.kickoff_utc))}</span>
          <span class="flex items-center gap-1.5">${gw}${evBadge}</span>
        </div>
        <div class="space-y-2">${row("home", f.home)}${row("away", f.away)}</div>
        <div>
          <div class="prob-bar" role="img" aria-label="Home ${pct(p.home)}, draw ${pct(p.draw)}, away ${pct(p.away)}">
            <span style="width:${p.home * 100}%;background:${COLORS.home}"></span>
            <span style="width:${p.draw * 100}%;background:${COLORS.draw}"></span>
            <span style="width:${p.away * 100}%;background:${COLORS.away}"></span>
          </div>
          <div class="mt-1.5 flex justify-between text-[11px] text-slate-500 tabular-nums">
            <span>Home ${pct(p.home)}</span><span class="${best === "draw" ? "text-slate-200 font-semibold" : ""}">Draw ${pct(p.draw)}</span><span>Away ${pct(p.away)}</span>
          </div>
        </div>
        <div class="flex flex-wrap gap-1.5">
          <span class="chip" title="Most likely exact score from 10,000 simulations">Most likely ${esc(bo.score)} · ${pct(bo.prob)}</span>
          <span class="chip" title="Expected goals (Poisson rates)">xG ${num(f.lambda_home, 1)}–${num(f.lambda_away, 1)}</span>
          <span class="chip">O2.5 ${pct(f.over25)}</span>
          <span class="chip">BTTS ${pct(f.btts)}</span>
          ${warn}
        </div>
        <div class="flex items-center justify-between text-xs">
          <span class="text-slate-400">Pick <span class="text-slate-100 font-medium">${esc(f.pick.label)}</span></span>
          <span class="text-slate-500">${esc(f.pick.confidence)} confidence →</span>
        </div>
      </button>`;
  }

  async function openFixture(id) {
    openModal("Loading…", "");
    try {
      renderDeepDive(await api(`/api/fixtures/${encodeURIComponent(id)}`));
    } catch (err) {
      $("#modal-body").innerHTML = `<p class="text-rose-300 text-sm">${esc(err.message)}</p>`;
    }
  }

  // ------------------------------------------------------------------ modal
  function openModal(title, sub) {
    $("#modal-title").textContent = title;
    $("#modal-sub").textContent = sub;
    $("#modal-body").innerHTML = '<div class="grid gap-4 lg:grid-cols-2"><div class="skeleton h-64"></div><div class="skeleton h-64"></div></div>';
    $("#modal").classList.remove("hidden");
    document.body.style.overflow = "hidden";
  }

  function closeModal() {
    $("#modal").classList.add("hidden");
    document.body.style.overflow = "";
    ["dd-probs", "dd-radar", "dd-heat", "dd-shap"].forEach((k) => { if (S.charts[k]) { S.charts[k].destroy(); delete S.charts[k]; } });
  }

  function renderDeepDive(d) {
    const p = d.probs;
    const sim = d.simulation;
    const when = d.kickoff_utc ? kickoff(d.kickoff_utc) : `As of ${dateOnly(d.date)}`;
    $("#modal-title").textContent = `${d.home} v ${d.away}`;
    $("#modal-sub").textContent = `${when}${d.gameweek ? ` · Gameweek ${d.gameweek}` : ""} · ${sim.n_sims.toLocaleString()} simulations`;

    const favoured = d.explain.favoured;
    const outcomeName = { home: `${d.home} win`, draw: "Draw", away: `${d.away} win` };
    const bigProb = (side, label) => `
      <div class="rounded-xl p-3 ring-1 ${favoured === side ? "ring-slate-600 bg-slate-800/60" : "ring-slate-800"}">
        <div class="text-[11px] uppercase tracking-wider" style="color:${COLORS[side]}">${esc(label)}</div>
        <div class="mt-1 text-2xl font-semibold text-slate-50 tabular-nums">${pct(p[side], 1)}</div>
        <div class="text-[11px] text-slate-500 tabular-nums">fair odds ${p[side] > 0 ? (1 / p[side]).toFixed(2) : "—"}</div>
      </div>`;

    $("#modal-body").innerHTML = `
      <div class="grid gap-4 lg:grid-cols-12">
        <section class="dd-card lg:col-span-5">
          <div class="dd-title">1X2 probabilities</div>
          <div class="grid grid-cols-3 gap-2">${bigProb("home", "Home")}${bigProb("draw", "Draw")}${bigProb("away", "Away")}</div>
          <div class="mt-4 h-40"><canvas id="dd-probs" aria-label="Probability comparison chart"></canvas></div>
          <p class="mt-3 text-sm text-slate-300">Pick: <span class="font-semibold text-slate-50">${esc(d.pick.label)}</span>
            <span class="text-slate-500">(${esc(d.pick.confidence)} confidence)</span></p>
          <div class="mt-2 flex flex-wrap gap-1.5">${styleChip(d.styles.home)}<span class="text-xs text-slate-500">vs</span>${styleChip(d.styles.away)}</div>
        </section>

        <section class="dd-card lg:col-span-7">
          <div class="dd-title">Team profiles <span class="normal-case tracking-normal font-normal text-slate-500">(percentile vs. all PL sides, last 5 matches)</span></div>
          <div class="h-72"><canvas id="dd-radar" aria-label="Radar chart comparing team profiles"></canvas></div>
        </section>

        <section class="dd-card lg:col-span-7">
          <div class="dd-title">Scoreline probabilities <span class="normal-case tracking-normal font-normal text-slate-500">(${d.home_short} goals ↓ · ${d.away_short} goals →)</span></div>
          <div class="h-80" id="dd-heat-wrap"><canvas id="dd-heat" aria-label="Scoreline heatmap"></canvas></div>
        </section>

        <section class="dd-card lg:col-span-5">
          <div class="dd-title">Goals markets</div>
          ${marketsHtml(d)}
        </section>

        <section class="dd-card lg:col-span-7">
          <div class="dd-title">Why the model leans ${esc(outcomeName[favoured])} <span class="normal-case tracking-normal font-normal text-slate-500">(SHAP, percentage points)</span></div>
          ${shapHtml(d)}
          <div class="mt-3 h-64"><canvas id="dd-shap" aria-label="SHAP driver chart"></canvas></div>
        </section>

        <section class="dd-card lg:col-span-5">
          <div class="dd-title">Value check</div>
          ${valueHtml(d)}
        </section>

        <section class="dd-card lg:col-span-7">
          <div class="dd-title">Squad availability <span class="normal-case tracking-normal font-normal text-slate-500">(FPL injury &amp; suspension news)</span></div>
          ${availabilityHtml(d)}
        </section>

        <section class="dd-card lg:col-span-5">
          <div class="dd-title">Pre-match numbers</div>
          ${featuresHtml(d)}
        </section>
      </div>`;

    drawProbChart(d);
    drawRadar(d);
    drawHeatmap(d);
    drawShap(d);
  }

  function marketsHtml(d) {
    const sim = d.simulation;
    const ou = Object.entries(sim.over_under).map(([line, v]) =>
      `<tr><td>${esc(line)}</td><td class="num">${pct(v.over, 1)}</td><td class="num">${pct(v.under, 1)}</td></tr>`).join("");
    const top = sim.top_scores.slice(0, 6).map((s, i) =>
      `<tr class="${i === 0 ? "hl" : ""}"><td>${esc(s.score)}</td><td class="num">${pct(s.prob, 1)}</td></tr>`).join("");
    const bbo = sim.best_by_outcome;
    return `
      <div class="grid grid-cols-2 gap-3 text-sm">
        <div class="rounded-lg bg-slate-800/40 p-2.5"><div class="text-[11px] text-slate-500">Expected goals</div>
          <div class="font-semibold text-slate-100 tabular-nums">${num(sim.lambda_home)} – ${num(sim.lambda_away)}</div></div>
        <div class="rounded-lg bg-slate-800/40 p-2.5"><div class="text-[11px] text-slate-500">Both teams score</div>
          <div class="font-semibold text-slate-100 tabular-nums">${pct(sim.btts.yes, 1)}</div></div>
        <div class="rounded-lg bg-slate-800/40 p-2.5"><div class="text-[11px] text-slate-500">${esc(d.home_short)} clean sheet</div>
          <div class="font-semibold text-slate-100 tabular-nums">${pct(sim.clean_sheet.home, 1)}</div></div>
        <div class="rounded-lg bg-slate-800/40 p-2.5"><div class="text-[11px] text-slate-500">${esc(d.away_short)} clean sheet</div>
          <div class="font-semibold text-slate-100 tabular-nums">${pct(sim.clean_sheet.away, 1)}</div></div>
      </div>
      <div class="mt-3 grid grid-cols-2 gap-4">
        <div class="overflow-x-auto"><table class="data-table"><thead><tr><th>Goals</th><th class="num">Over</th><th class="num">Under</th></tr></thead><tbody>${ou}</tbody></table></div>
        <div class="overflow-x-auto"><table class="data-table"><thead><tr><th>Score</th><th class="num">Prob.</th></tr></thead><tbody>${top}</tbody></table></div>
      </div>
      <p class="mt-3 text-xs text-slate-500">Most likely by result: home win ${esc(bbo.home?.score ?? "—")}, draw ${esc(bbo.draw?.score ?? "—")}, away win ${esc(bbo.away?.score ?? "—")}.
        Dixon–Coles ρ = ${num(sim.rho, 3)}.</p>`;
  }

  function shapHtml(d) {
    const e = d.explain;
    const key = e.favoured;
    const base = e.baseline[key], raw = e.model_raw[key], fin = e.final[key];
    const cal = e.calibration_adjustment[key];
    const lines = (e.summary || []).map((s) => `<li>${esc(s)}</li>`).join("");
    return `
      <div class="grid sm:grid-cols-3 gap-2 text-xs">
        <div class="rounded-lg bg-slate-800/40 p-2"><div class="text-slate-500">Baseline (avg. fixture, incl. home advantage)</div><div class="text-slate-100 font-semibold tabular-nums">${pct(base, 1)}</div></div>
        <div class="rounded-lg bg-slate-800/40 p-2"><div class="text-slate-500">Model after drivers · calibration</div><div class="text-slate-100 font-semibold tabular-nums">${pct(raw, 1)} · ${signed(cal)} pts</div></div>
        <div class="rounded-lg bg-slate-800/40 p-2"><div class="text-slate-500">Final probability</div><div class="text-slate-100 font-semibold tabular-nums">${pct(fin, 1)}</div></div>
      </div>
      ${lines ? `<ul class="mt-3 list-disc pl-5 text-sm text-slate-300 space-y-0.5">${lines}</ul>` : ""}
      <p class="mt-2 text-[11px] text-slate-500">${esc(e.backend)}. Bars show each driver's push on the ${esc(key)} probability; related features are grouped.</p>`;
  }

  function valueHtml(d) {
    const v = d.value;
    if (!v.odds || !Object.keys(v.odds).length) {
      return `<p class="text-sm text-slate-400">No bookmaker prices for this fixture yet. football-data.co.uk publishes Bet365 odds a few days before kickoff.
        You can enter prices in the <a href="#simulator" class="text-sky-400 hover:underline" data-close>simulator</a> to run the EV engine now.</p>`;
    }
    const rows = v.selections.map((s) => `
      <tr class="${s.is_value ? "hl" : ""}">
        <td>${esc(s.label)}</td><td class="num">${num(s.odds)}</td><td class="num">${pct(s.model_prob, 1)}</td>
        <td class="num">${pct(s.fair_prob, 1)}</td>
        <td class="num ${s.ev > 0 ? "text-emerald-300" : "text-slate-400"}">${signed(s.ev * 100)}%</td>
        <td>${s.is_value ? `<span class="ev-badge ${s.longshot ? "caution" : ""}">${s.longshot ? "+EV longshot" : "+EV"}</span>` : ""}</td>
      </tr>`).join("");
    const margins = Object.entries(v.markets).map(([m, x]) => `${esc(m)} margin ${pct(x.margin, 1)}`).join(" · ");
    return `
      <div class="overflow-x-auto"><table class="data-table"><thead><tr><th>Selection</th><th class="num">Odds</th><th class="num">Model</th><th class="num">Fair</th><th class="num">EV</th><th></th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="mt-2 text-[11px] text-slate-500">EV = model probability × odds − 1; flagged above ${Math.round(v.threshold * 100)}%. “Fair” removes the bookmaker margin (Shin method). ${margins}. Source: ${esc(v.odds_source)}.</p>
      <p class="mt-1 text-[11px] text-amber-300/80">Backtested flags have not beaten Bet365 so far: see Model validation. Longshot flags (odds &gt; 5) have been the worst.</p>`;
  }

  function availabilityHtml(d) {
    const a = d.availability;
    if (!a.applied) {
      return `<p class="text-sm text-slate-400">Not applied: FPL availability describes the next gameweek only${a.home || a.away ? "" : ", or no squad data is available"}.</p>`;
    }
    const side = (key, name) => {
      const r = a[key] || { missing: [], key_players: [], attack_penalty: 0, defence_penalty: 0 };
      const missing = r.missing.length ? r.missing.map((m) => `
        <li class="flex items-start gap-2">
          <span class="chip ${m.loss >= 0.5 ? "warn-chip" : ""}">${esc(m.position)}</span>
          <span class="min-w-0"><span class="text-slate-100">${esc(m.web_name || m.name)}</span>
            <span class="text-slate-500">· ${m.chance == null ? esc(m.status) : `${m.chance}% chance`}${m.attack_share >= 0.02 ? ` · ${pct(m.attack_share)} of attack` : ""}</span>
            ${m.news ? `<span class="block text-[11px] text-slate-500 truncate" title="${esc(m.news)}">${esc(m.news)}</span>` : ""}</span>
        </li>`).join("") : '<li class="text-slate-500">No notable absences</li>';
      const keys = (r.key_players || []).map((k) => `${esc(k.name)} ${pct(k.attack_share)}`).join(", ");
      return `
        <div>
          <div class="flex items-baseline justify-between"><span class="font-semibold text-slate-100">${esc(name)}</span>
            <span class="text-xs tabular-nums ${r.attack_penalty > 0.03 ? "text-amber-300" : "text-slate-500"}">attack −${pct(r.attack_penalty, 1)} · defence +${pct(r.defence_penalty, 1)}</span></div>
          <ul class="mt-2 space-y-1.5 text-sm">${missing}</ul>
          <p class="mt-2 text-[11px] text-slate-500">Key attackers: ${keys || "—"}</p>
        </div>`;
    };
    return `<div class="grid sm:grid-cols-2 gap-5">${side("home", d.home)}${side("away", d.away)}</div>
      <p class="mt-3 text-[11px] text-slate-500">Expected goals × ${num(a.multipliers.home, 3)} (${esc(d.home_short)}) and × ${num(a.multipliers.away, 3)} (${esc(d.away_short)}). Weighted by each player's share of this season's attacking output and minutes.</p>`;
  }

  function featuresHtml(d) {
    const intish = new Set(["elo", "rest", "spell_games"]);
    const rows = d.features.map((f) => {
      const fmt = (x) => (intish.has(f.key) ? num(x, 0) : f.key === "pass_share5" ? pct(x, 0) : num(x, 2));
      return `<tr><td>${esc(f.label)}</td><td class="num">${fmt(f.home)}</td><td class="num">${fmt(f.away)}</td></tr>`;
    }).join("");
    return `<div class="max-h-80 overflow-y-auto"><div class="overflow-x-auto"><table class="data-table"><thead><tr><th>Metric</th><th class="num">${esc(d.home_short)}</th><th class="num">${esc(d.away_short)}</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
  }

  // ------------------------------------------------------------------ charts
  function drawProbChart(d) {
    const rows = [["Final", d.probs], ["Model (pre-availability)", d.probs_model], ["Poisson simulation", d.simulation.probs]];
    const fair = d.value.markets && d.value.markets["1X2"];
    if (fair) rows.push(["Market (fair)", fair.fair]);
    const ds = (key, label) => ({ label, data: rows.map((r) => r[1][key] * 100), backgroundColor: COLORS[key], borderWidth: 0, barThickness: 16 });
    chart("dd-probs", $("#dd-probs"), {
      type: "bar",
      data: { labels: rows.map((r) => r[0]), datasets: [ds("home", "Home"), ds("draw", "Draw"), ds("away", "Away")] },
      options: {
        indexAxis: "y", responsive: true, maintainAspectRatio: false,
        scales: { x: { stacked: true, max: 100, ticks: { callback: (v) => `${v}%` }, grid: { color: COLORS.grid } }, y: { stacked: true, grid: { display: false } } },
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.x.toFixed(1)}%` } } },
      },
    });
  }

  function drawRadar(d) {
    const r = d.radar;
    const ds = (key, name) => ({
      label: name, data: r[key], borderColor: COLORS[key], backgroundColor: `${COLORS[key]}33`, pointBackgroundColor: COLORS[key], borderWidth: 2, pointRadius: 3,
    });
    chart("dd-radar", $("#dd-radar"), {
      type: "radar",
      data: { labels: r.axes, datasets: [ds("home", d.home), ds("away", d.away)] },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: { r: { min: 0, max: 100, ticks: { display: false, stepSize: 25 }, grid: { color: COLORS.grid }, angleLines: { color: COLORS.grid }, pointLabels: { color: "#cbd5e1", font: { size: 11 } } } },
        plugins: {
          legend: { position: "bottom", labels: { boxWidth: 10 } },
          tooltip: { callbacks: { label: (c) => {
            const side = c.datasetIndex === 0 ? "home" : "away";
            return `${c.dataset.label}: ${c.parsed.r.toFixed(0)}th pct (value ${r.raw[side][c.dataIndex]})`;
          } } },
        },
      },
    });
  }

  function drawHeatmap(d) {
    const m = d.simulation.matrix;
    const labels = d.simulation.matrix_labels;
    const max = Math.max(...m.flat());
    const hasMatrix = !!(Chart.registry && (() => { try { return Chart.registry.getController("matrix"); } catch (_) { return null; } })());
    if (!hasMatrix) {  // plugin failed to load: plain HTML grid
      const cells = m.map((row, i) => row.map((v, j) => `<div class="heat-cell" style="background:rgba(56,189,248,${(v / max) * 0.85 + 0.05})" title="${labels[i]}-${labels[j]}">${(v * 100).toFixed(1)}</div>`).join("")).join("");
      $("#dd-heat-wrap").innerHTML = `<div class="heat-grid" style="grid-template-columns:repeat(${labels.length},1fr)">${cells}</div>`;
      return;
    }
    const data = [];
    m.forEach((row, i) => row.forEach((v, j) => data.push({ x: labels[j], y: labels[i], v })));
    chart("dd-heat", $("#dd-heat"), {
      type: "matrix",
      data: { datasets: [{
        label: "P(score)", data,
        backgroundColor: (c) => { const v = c.dataset.data[c.dataIndex]?.v ?? 0; return `rgba(56,189,248,${(v / max) * 0.9 + 0.04})`; },
        borderColor: "rgba(15,23,42,0.9)", borderWidth: 1,
        width: ({ chart: ch }) => (ch.chartArea || {}).width / labels.length - 2,
        height: ({ chart: ch }) => (ch.chartArea || {}).height / labels.length - 2,
      }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { type: "category", labels, offset: true, position: "top", grid: { display: false }, title: { display: true, text: `${d.away_short} goals` } },
          y: { type: "category", labels, offset: true, reverse: false, grid: { display: false }, title: { display: true, text: `${d.home_short} goals` } },
        },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { title: () => "", label: (c) => `${d.home_short} ${c.raw.y} – ${c.raw.x} ${d.away_short}: ${(c.raw.v * 100).toFixed(1)}%` } },
        },
      },
      plugins: [{
        id: "cellLabels",
        afterDatasetsDraw(ch) {
          const { ctx } = ch;
          const meta = ch.getDatasetMeta(0);
          ctx.save();
          ctx.font = "10px Inter, sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          meta.data.forEach((el, idx) => {
            const v = data[idx].v;
            if (v < 0.005) return;
            const { x, y, width, height } = el.getProps(["x", "y", "width", "height"], true);
            ctx.fillStyle = v / max > 0.55 ? "#0f172a" : "#e2e8f0";
            ctx.fillText(`${(v * 100).toFixed(1)}`, x + width / 2, y + height / 2);
          });
          ctx.restore();
        },
      }],
    });
  }

  function drawShap(d) {
    const e = d.explain;
    const key = e.favoured;
    const drivers = e.drivers.slice(0, 8);
    chart("dd-shap", $("#dd-shap"), {
      type: "bar",
      data: {
        labels: drivers.map((x) => x.label),
        datasets: [{ data: drivers.map((x) => x[key]), backgroundColor: drivers.map((x) => (x[key] >= 0 ? "#34d399" : "#fb7185")), borderWidth: 0, barThickness: 14 }],
      },
      options: {
        indexAxis: "y", responsive: true, maintainAspectRatio: false,
        scales: { x: { ticks: { callback: (v) => `${v > 0 ? "+" : ""}${v}` }, title: { display: true, text: `pts toward ${key} win`.replace("draw win", "draw") }, grid: { color: COLORS.grid } },
                  y: { grid: { display: false }, ticks: { color: "#cbd5e1" } } },
        plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => {
          const x = drivers[c.dataIndex];
          return `home ${signed(x.home)} · draw ${signed(x.draw)} · away ${signed(x.away)} pts`;
        } } } },
      },
    });
  }

  // ------------------------------------------------------------------ simulator
  async function loadTeams(includeAll = false) {
    try {
      const teams = await api(`/api/teams${includeAll ? "?all=true" : ""}`);
      if (includeAll) S.allTeams = teams; else S.teams = teams;
      fillTeamSelects(teams);
    } catch (err) {
      toast(`Could not load teams: ${err.message}`, "err");
    }
  }

  function fillTeamSelects(teams) {
    const home = $("#sim-home"), away = $("#sim-away");
    const prevH = home.value, prevA = away.value;
    const opts = teams.map((t) => `<option value="${esc(t.name)}">${esc(t.name)}${t.in_current_season ? "" : " (not in PL)"} · ${esc(t.style)}</option>`).join("");
    home.innerHTML = opts;
    away.innerHTML = opts;
    home.value = prevH || (teams.find((t) => t.name === "Liverpool") || teams[0]).name;
    away.value = prevA || (teams.find((t) => t.name === "Manchester City") || teams[1] || teams[0]).name;
  }

  async function simulate(ev) {
    ev.preventDefault();
    const form = ev.target;
    const err = $("#sim-error");
    err.classList.add("hidden");
    const body = { home: $("#sim-home").value, away: $("#sim-away").value, apply_availability: $("#sim-avail").checked };
    if (body.home === body.away) {
      err.textContent = "Pick two different clubs.";
      err.classList.remove("hidden");
      return;
    }
    if ($("#sim-date").value) body.date = $("#sim-date").value;
    const odds = {};
    ["home", "draw", "away", "over25", "under25"].forEach((k) => {
      const v = parseFloat(form.elements[`odds_${k}`].value);
      if (v > 1) odds[k] = v;
    });
    if (Object.keys(odds).length) body.odds = odds;
    const btn = $("#sim-submit");
    btn.disabled = true;
    btn.textContent = "Simulating…";
    openModal(`${body.home} v ${body.away}`, "Running 10,000 simulations…");
    try {
      renderDeepDive(await api("/api/simulate", { method: "POST", body: JSON.stringify(body) }));
    } catch (e) {
      closeModal();
      const hint = e.body && e.body.suggestions && e.body.suggestions.length ? ` Did you mean ${e.body.suggestions.join(", ")}?` : "";
      err.textContent = `${e.message}${hint}`;
      err.classList.remove("hidden");
    } finally {
      btn.disabled = false;
      btn.textContent = "Simulate";
    }
  }

  // ------------------------------------------------------------------ model panel
  async function loadModel() {
    try {
      renderModel(await api("/api/metrics"));
    } catch (err) {
      $("#model-panel").innerHTML = `<div class="panel p-6 text-sm text-rose-300">Could not load metrics: ${esc(err.message)}</div>`;
    }
  }

  function metricRows(block, order, labels, live) {
    return order.filter((k) => block[k]).map((k) => {
      const m = block[k];
      return `<tr class="${k === live ? "hl" : ""}"><td>${esc(labels[k] || k)}</td><td class="num">${m.n}</td><td class="num">${num(m.log_loss, 4)}</td>
        <td class="num">${num(m.brier, 4)}</td><td class="num">${m.rps != null ? num(m.rps, 4) : "—"}</td><td class="num">${pct(m.accuracy, 1)}</td></tr>`;
    }).join("");
  }

  function renderModel(v) {
    const labels = {
      base_rate: "Base rate", bet365: "Bet365 (margin removed)", bookmakers: "Market avg closing odds",
      random_forest: "RandomForest baseline", xgboost: "XGBoost uncalibrated", xgboost_calibrated: "XGBoost + calibration (live)",
      poisson: "Poisson simulation 1X2", xgboost_calibrated_on_odds_subset: "Live model, rows with odds", classifier: "XGBoost + calibration (live)",
    };
    const head = '<thead><tr><th>Model</th><th class="num">n</th><th class="num">Log loss</th><th class="num">Brier</th><th class="num">RPS</th><th class="num">Acc.</th></tr></thead>';
    const hold = metricRows(v.holdout, ["base_rate", "bet365", "bookmakers", "random_forest", "xgboost", "poisson", "xgboost_calibrated"], labels, "xgboost_calibrated");
    const cur = v.current_season ? metricRows(v.current_season, ["base_rate", "bet365", "xgboost_calibrated"], labels, "xgboost_calibrated") : "";
    const bt = v.backtest || {};
    const btRows = bt.classifier ? metricRows(bt, ["base_rate", "bet365", "classifier"], labels, "classifier") : "";
    const vb = (bt.value_bets || {});
    const betLine = (name, r) => r && r.bets ? `
      <tr><td>${esc(name)}</td><td class="num">${r.bets}</td><td class="num">${pct(r.hit_rate, 0)}</td><td class="num">${num(r.avg_odds)}</td>
        <td class="num ${r.roi >= 0 ? "text-emerald-300" : "text-rose-300"}">${signed(r.roi * 100)}% ± ${num(r.roi_se * 100, 1)}</td></tr>` : "";
    const bands = ((vb["1x2"] || {}).bands || []).map((b) => `<span class="chip">odds ${esc(b.odds)}: ${b.bets} bets, ${signed(b.roi * 100)}%</span>`).join(" ");
    const g = v.goals || {};
    const ou = (bt.over25 || {});
    const tactics = (v.tactics || []).map((t) => {
      const clubs = S.teams.filter((x) => x.style === t.style).map((x) => x.name).join(", ");
      return `<tr><td>${styleChip(t.style)}</td><td class="num">${num(t.ppda, 1)}</td><td class="num">${pct(t.pass_share, 0)}</td><td class="num">${num(t.directness, 1)}</td></tr>
        ${clubs ? `<tr><td colspan="4" class="text-[11px] text-slate-500">${esc(clubs)}</td></tr>` : ""}`;
    }).join("");
    const cm = v.holdout_confusion || [];
    const cmRows = ["Home", "Draw", "Away"].map((l, i) => `<tr><td>${l}</td>${(cm[i] || []).map((x) => `<td class="num">${x}</td>`).join("")}</tr>`).join("");

    $("#model-panel").innerHTML = `
      <div class="panel p-5">
        <div class="dd-title">Hold-out season ${esc(v.holdout_season)} <span class="normal-case tracking-normal font-normal text-slate-500">(trained ${esc(v.holdout_train_seasons[0])} → ${esc(v.holdout_train_seasons.at(-1))})</span></div>
        <div class="overflow-x-auto"><table class="data-table">${head}<tbody>${hold}</tbody></table></div>
        ${cur ? `<div class="dd-title mt-5">Current season ${esc(v.current_season_label)} so far</div><div class="overflow-x-auto"><table class="data-table">${head}<tbody>${cur}</tbody></table></div>` : ""}
      </div>
      <div class="panel p-5">
        <div class="dd-title">Walk-forward backtest ${esc((bt.seasons || []).join(", "))} <span class="normal-case tracking-normal font-normal text-slate-500">(${bt.matches || 0} matches)</span></div>
        <div class="overflow-x-auto"><table class="data-table">${head}<tbody>${btRows}</tbody></table></div>
        <div class="dd-title mt-5">Value flags (EV &gt; ${Math.round((vb.threshold || 0.05) * 100)}%) vs Bet365, flat 1-unit stakes</div>
        <div class="overflow-x-auto"><table class="data-table"><thead><tr><th>Market</th><th class="num">Bets</th><th class="num">Hit</th><th class="num">Avg odds</th><th class="num">ROI ± SE</th></tr></thead>
          <tbody>${betLine("1X2 (calibrated classifier)", vb["1x2"])}${betLine("Over/Under 2.5 (Poisson)", vb.over_under_2_5)}</tbody></table></div>
        <div class="mt-2 flex flex-wrap gap-1.5">${bands}</div>
        <p class="mt-2 text-[11px] text-amber-300/80">No demonstrated edge: ROI is within noise of zero or negative after the bookmaker margin. Treat badges as model-vs-market disagreements, not advice.</p>
      </div>
      <div class="panel p-5">
        <div class="dd-title">Goal model (Poisson regressors + Dixon–Coles)</div>
        <div class="grid grid-cols-2 gap-3 text-sm">
          <div class="rounded-lg bg-slate-800/40 p-2.5"><div class="text-[11px] text-slate-500">Mean λ vs actual goals (${esc(v.holdout_season)})</div>
            <div class="font-semibold text-slate-100 tabular-nums">${num(g.mean_lambda?.home)}–${num(g.mean_lambda?.away)} vs ${num(g.mean_goals?.home)}–${num(g.mean_goals?.away)}</div></div>
          <div class="rounded-lg bg-slate-800/40 p-2.5"><div class="text-[11px] text-slate-500">Poisson deviance (home, vs constant)</div>
            <div class="font-semibold text-slate-100 tabular-nums">${num(g.poisson_deviance?.home, 3)} vs ${num(g.poisson_deviance?.home_baseline, 3)}</div></div>
          <div class="rounded-lg bg-slate-800/40 p-2.5"><div class="text-[11px] text-slate-500">Over 2.5 Brier, 3 seasons (model · base · Bet365)</div>
            <div class="font-semibold text-slate-100 tabular-nums">${num(ou.model?.brier, 4)} · ${num(ou.base_rate?.brier, 4)} · ${num(ou.bet365?.brier, 4)}</div></div>
          <div class="rounded-lg bg-slate-800/40 p-2.5"><div class="text-[11px] text-slate-500">Dixon–Coles ρ</div>
            <div class="font-semibold text-slate-100 tabular-nums">${num(g.rho, 4)}</div></div>
        </div>
        <p class="mt-3 text-[11px] text-slate-500">The goal model ranks attacking and defensive strength well (its 1X2 is competitive with the classifier), but total-goals levels drift between seasons, so Over/Under probabilities are only as good as the base rate.</p>
      </div>
      <div class="panel p-5">
        <div class="dd-title">Tactical archetypes (k-means on rolling PPDA, possession share, directness)</div>
        <div class="overflow-x-auto"><table class="data-table"><thead><tr><th>Style</th><th class="num">PPDA</th><th class="num">Poss.</th><th class="num">Deep/100 passes</th></tr></thead><tbody>${tactics}</tbody></table></div>
        <div class="dd-title mt-5">Confusion matrix (hold-out, rows = actual)</div>
        <div class="overflow-x-auto"><table class="data-table w-auto"><thead><tr><th></th><th class="num">Home</th><th class="num">Draw</th><th class="num">Away</th></tr></thead><tbody>${cmRows}</tbody></table></div>
      </div>
      ${(v.plots || []).length ? `<div class="panel p-5 lg:col-span-2"><div class="dd-title">Calibration (hold-out season)</div>
        <img src="/reports/calibration_holdout.png" alt="Calibration curves for home, draw and away probabilities" class="w-full rounded-lg bg-white" loading="lazy"></div>` : ""}`;
  }

  // ------------------------------------------------------------------ sync
  async function triggerSync() {
    if (S.syncing) return;
    let token = sessionStorage.getItem("adminToken") || "";
    const send = () => api("/api/sync", { method: "POST", body: JSON.stringify({ retrain: false }), headers: token ? { "X-Admin-Token": token } : {} });
    try {
      await send();
    } catch (err) {
      if (err.status === 401) {
        token = window.prompt("Admin token required to sync:") || "";
        if (!token) return;
        try { await send(); sessionStorage.setItem("adminToken", token); } catch (e) { toast(e.message, "err"); return; }
      } else if (err.status !== 409) {
        toast(`Sync failed to start: ${err.message}`, "err");
        return;
      }
    }
    setSyncing(true);
    toast("Syncing football-data, Understat and FPL…", "info", 3000);
    pollSync();
  }

  function setSyncing(on) {
    S.syncing = on;
    $("#sync-btn").disabled = on;
    $("#sync-label").textContent = on ? "Syncing…" : "Sync data";
    $("#sync-btn").setAttribute("aria-label", on ? "Syncing" : "Sync data");
    $("#sync-icon").classList.toggle("spin", on);
    if (on) renderPill("syncing", "Syncing…");
  }

  async function pollSync() {
    try {
      const st = await api("/api/sync/status");
      if (st.state === "running") { setTimeout(pollSync, 2000); return; }
      setSyncing(false);
      if (st.state === "succeeded") {
        const r = st.result || {};
        const fresh = r.new_results ? `${r.new_results} new result(s)` : "No new results";
        toast(`${fresh}. Model ${r.retrained ? "retrained" : "unchanged"}${r.failures ? ` · ${r.failures} source(s) failed` : ""}.`, r.failures ? "info" : "ok");
        S.ready = false;
        pollStatus();
      } else if (st.state === "failed") {
        toast(`Sync failed: ${st.error}`, "err", 8000);
        pollStatus();
      }
    } catch (err) {
      setTimeout(pollSync, 4000);
    }
  }

  // ------------------------------------------------------------------ wiring
  document.addEventListener("DOMContentLoaded", () => {
    $("#sync-btn").addEventListener("click", triggerSync);
    $("#sim-form").addEventListener("submit", simulate);
    $("#sim-swap").addEventListener("click", () => {
      const h = $("#sim-home"), a = $("#sim-away");
      [h.value, a.value] = [a.value, h.value];
    });
    $("#sim-all").addEventListener("change", async (e) => {
      if (e.target.checked) { if (!S.allTeams) await loadTeams(true); else fillTeamSelects(S.allTeams); }
      else fillTeamSelects(S.teams);
    });
    $$(".seg-btn").forEach((b) => b.addEventListener("click", () => {
      $$(".seg-btn").forEach((x) => x.classList.remove("seg-active"));
      b.classList.add("seg-active");
      S.days = b.dataset.days;
      if (S.ready) loadFixtures();
    }));
    $("#modal").addEventListener("click", (e) => { if (e.target.closest("[data-close]")) closeModal(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#modal").classList.contains("hidden")) closeModal(); });
    pollStatus();
    // Keep "last sync"/"next sync" fresh and pick up scheduled syncs.
    setInterval(async () => { if (S.ready && !S.syncing) { try { S.status = await api("/api/status"); renderStatus(S.status); } catch (_) {} } }, 60000);
  });
})();
