import React, { useCallback, useEffect, useState } from 'react'
import { getDashboardData, patchLabel, predictTransaction, predictBatch } from './api/dashboardApi.js'
import RiskOrb from './components/RiskOrb.jsx'
import CountUp from './components/CountUp.jsx'
import ThemeToggle from './components/ThemeToggle.jsx'
import MotionToggle from './components/MotionToggle.jsx'
import PredictionScatter from './three/PredictionScatter.jsx'
import ConfusionBars from './three/ConfusionBars.jsx'

const NAV_ITEMS = [
  { id: 'overview', label: 'Overview', group: 'WORKSPACE' },
  { id: 'predict', label: 'Risk assessment', group: 'WORKSPACE' },
  { id: 'batch', label: 'Batch scoring', group: 'WORKSPACE' },
  { id: 'monitoring', label: 'Monitoring', group: 'INTELLIGENCE' },
  { id: 'model', label: 'Model registry', group: 'INTELLIGENCE' },
  { id: 'history', label: 'Transactions', group: 'INTELLIGENCE' },
]
const NAV_ICONS = {
  overview: <><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>,
  predict: <><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3" /><path d="M7 12h10M12 7v10" /></>,
  batch: <><path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 12 9 5 9-5M3 16l9 5 9-5" /></>,
  monitoring: <path d="M3 12h4l3-8 4 16 3-8h4" />,
  model: <><path d="m12 3 9 5v8l-9 5-9-5V8l9-5Z" /><path d="m3 8 9 5 9-5M12 13v8" /></>,
  history: <><path d="M9 6h11M9 12h11M9 18h11" /><path d="M4 6h.01M4 12h.01M4 18h.01" /></>,
}
const TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
const EMPTY_FORM = { step: 1, type: 'PAYMENT', amount: '', oldbalanceOrg: '', newbalanceOrig: '', oldbalanceDest: '', newbalanceDest: '' }
const PRESETS = {
  safe: { step: 240, type: 'PAYMENT', amount: 825, oldbalanceOrg: 5200, newbalanceOrig: 4375, oldbalanceDest: 0, newbalanceDest: 825 },
  elevated: { step: 420, type: 'TRANSFER', amount: 985000, oldbalanceOrg: 985000, newbalanceOrig: 0, oldbalanceDest: 12000, newbalanceDest: 12000 },
}

function usePage() {
  const read = () => window.location.hash.replace('#', '') || 'overview'
  const [page, setPage] = useState(read)
  useEffect(() => {
    const onHashChange = () => setPage(read())
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])
  return page
}

function formatNumber(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—'
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits })
}
function shortId(id = '') { return id ? `${id.slice(0, 8)}…` : '…' }

function NavIcon({ id }) {
  return <span className="nav-icon" aria-hidden="true">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" focusable="false">
      {NAV_ICONS[id]}
    </svg>
  </span>
}

function Sidebar({ page }) {
  const groups = [...new Set(NAV_ITEMS.map((item) => item.group))]
  return <aside className="sidebar">
    <a className="brand" href="#overview"><span className="brand-symbol" aria-hidden="true"><svg viewBox="0 0 28 28" fill="none"><path d="M5 15h4l2.2-6 4.1 11 2.3-6H23" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/><circle cx="14" cy="14" r="11" stroke="currentColor" strokeOpacity=".35" strokeWidth="1.2"/></svg></span><span>Fraud Detection</span></a>
    <div className="workspace-switch"><span className="workspace-avatar">P</span><span><b>PaySim workspace</b><small>Fraud operations</small></span><span className="chevron" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="m7 10 5 5 5-5" /></svg></span></div>
    <nav aria-label="Main navigation">
      {groups.map((group) => <div className="nav-group" key={group}><p>{group}</p>{NAV_ITEMS.filter((item) => item.group === group).map((item) =>
        <a key={item.id} className={`nav-link ${page === item.id ? 'active' : ''}`} href={`#${item.id}`} aria-label={item.label} title={item.label}><NavIcon id={item.id} />{item.label}{item.id === 'monitoring' && <i className="nav-pulse" />}</a>)}</div>)}
    </nav>
    <div className="sidebar-bottom"><div className="secure-pill"><span aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Zm-3-10 2 2 4-4" /></svg></span> Secure inference active</div><div className="sidebar-version">FRAUD DETECTION CONSOLE <b>v2.4.0</b></div></div>
  </aside>
}

function Topbar({ title, health, theme, setTheme, reducedMotion, setReducedMotion }) {
  return <header className="topbar"><div className="breadcrumb">Fraud Detection <span>/</span> <b>{title}</b></div><div className="topbar-right">
    <div className={`api-status ${health?.model_loaded ? 'online' : 'offline'}`}><i />{health?.model_loaded ? 'MODEL ONLINE' : 'MODEL OFFLINE'}</div>
    <ThemeToggle theme={theme} onChange={setTheme} /><MotionToggle reduced={reducedMotion} onChange={setReducedMotion} />
    <button className="icon-button" title="Notifications" aria-label="Notifications"><span aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9Zm-8 13h4" /></svg></span><i /></button>
    <div className="top-user"><span className="user-avatar">RG</span><span><b>Ritesh Girhe</b><small>Administrator</small></span><span className="chevron" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="m7 10 5 5 5-5" /></svg></span></div>
  </div></header>
}

function PageTitle({ eyebrow, title, description, action }) {
  return <div className="page-title"><div><div className="eyebrow"><span />{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action}</div>
}

function StatCard({ label, value, note, icon, tone = '', loading = false }) {
  const symbol = icon === 'transactions' ? <><path d="M8 4h8l3 3v13H5V4h3Z"/><path d="M8 11h8M8 15h8"/></> : icon === 'fraud' ? <><path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z"/><path d="M12 8v5m0 3h.01"/></> : icon === 'latency' ? <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></> : <><path d="M12 3v18M5 8l7-5 7 5M5 16l7 5 7-5"/></>
  return <article className="stat-card" aria-busy={loading}><div className="stat-top"><span>{label}</span><i className={`stat-icon ${tone}`}><svg viewBox="0 0 24 24" aria-hidden="true">{symbol}</svg></i></div><strong className={loading ? 'skeleton' : ''}><CountUp value={value} /></strong><small>{note}</small></article>
}

function Sparkline({ points, color = '#64dfad' }) {
  if (!points?.length) return <div className="chart-empty">No prediction history yet</div>
  const max = Math.max(...points, 1)
  const coords = points.map((point, index) => `${(index / Math.max(points.length - 1, 1)) * 100},${100 - (point / max) * 86}`).join(' ')
  return <svg className="sparkline" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="Prediction volume trend"><defs><linearGradient id="spark-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor={color} stopOpacity=".24" /><stop offset="100%" stopColor={color} stopOpacity="0" /></linearGradient></defs><polygon points={`0,100 ${coords} 100,100`} fill="url(#spark-fill)" /><polyline points={coords} fill="none" stroke={color} strokeWidth="2" vectorEffect="non-scaling-stroke" /></svg>
}

function OverviewPage({ data, busy, error }) {
  const summary = data.summary || {}
  const predictions = data.predictions?.items || []
  const model = data.model || {}
  const bins = Array.from({ length: 20 }, (_, index) => predictions.filter((row) => Math.min(19, Math.floor(row.fraud_probability * 20)) === index).length)
  return <div className="page-content">
    <PageTitle eyebrow="PAYMENTS INTELLIGENCE / OVERVIEW" title={<>Good afternoon, <em>Ritesh.</em></>} description="Your fraud operations overview, from live transaction signals to model health." action={<div className="dashboard-actions"><a className="button button-primary" href="#predict">Score a transaction</a><a className="button button-secondary" href="#batch">Upload a batch</a><a className="button button-secondary" href="#history">Review history</a></div>} />
    {error && <div className="notice notice-error" role="alert">{error} <a href="#monitoring">Check service status</a></div>}
    <section className="stat-grid">
      <StatCard loading={busy && !data.health} label="Transactions scored" value={formatNumber(summary.predictions_stored)} note="Persisted prediction history" icon="transactions" />
      <StatCard loading={busy && !data.health} label="Flagged as fraud" value={formatNumber(summary.fraud_predictions_stored)} note={`${(100 * (summary.fraud_prediction_rate || 0)).toFixed(2)}% of scored transactions`} icon="fraud" tone="red" />
      <StatCard loading={busy && !data.health} label="P95 inference latency" value={`${formatNumber(summary.latency_p95_ms, 1)} ms`} note="Recent API request window" icon="latency" tone="blue" />
      <StatCard loading={busy && !data.health} label="Decision threshold" value={`${(100 * (model.threshold || 0)).toFixed(1)}%`} note={`Production ${model.version || 'model'}`} icon="threshold" tone="violet" />
    </section>
    <section className="overview-grid">
      <article className="panel trend-panel"><div className="panel-head"><div><span className="panel-kicker">LIVE SIGNAL</span><h2>Scoring activity</h2></div><span className="tag tag-live"><i /> LOCAL HISTORY</span></div><div className="chart-legend"><span><i aria-hidden="true" /> Probability distribution <span className="legend-separator" aria-hidden="true" /> recent transactions</span></div>
        {predictions.length ? <div className="histogram" role="img" aria-label={`Probability distribution for ${predictions.length} recent transactions`}>{bins.map((count, index) => <div className="histogram-col" key={index}><i style={{ height: `${count ? Math.max(8, count / Math.max(...bins, 1) * 100) : 2}%`, opacity: count ? .35 + index / 34 : .08 }} /></div>)}</div> : <div className="chart-empty"><span>No scored transactions yet</span><a className="text-link" href="#predict">Score your first transaction</a></div>}
        <div className="chart-axis"><span>0% risk</span><span>50%</span><span>100% risk</span></div><div className="chart-footer"><span>{predictions.length ? `Based on ${predictions.length} recent predictions` : 'Prediction history will appear here after your first assessment'}</span><a href="#history">View transactions</a></div>
      </article>
      <article className="panel model-panel"><div className="panel-head"><div><span className="panel-kicker">PRODUCTION MODEL</span><h2>Model health</h2></div><span className="model-version">{model.version || '—'}</span></div><div className="health-orbit"><div className="orbit-ring orbit-one" /><div className="orbit-ring orbit-two" /><div className="health-core"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12h4l3-8 4 16 3-8h4" /></svg></div><span className="orbit-label label-top">LIVE INFERENCE</span><span className="orbit-label label-bottom">{model.model || 'MODEL'}</span></div><div className="model-metrics"><div><span>PR-AUC</span><b>{model.metrics?.pr_auc == null ? '—' : Number(model.metrics.pr_auc).toFixed(4)}</b></div><div><span>Recall</span><b>{model.metrics?.recall == null ? '—' : `${(100 * model.metrics.recall).toFixed(1)}%`}</b></div><div><span>Precision</span><b>{model.metrics?.precision == null ? '—' : `${(100 * model.metrics.precision).toFixed(1)}%`}</b></div></div><a className="panel-link" href="#model">Inspect model details <span aria-hidden="true">→</span></a></article>
    </section>
    <section className="panel recent-panel"><div className="panel-head"><div><span className="panel-kicker">LATEST SIGNALS</span><h2>Recent transactions</h2></div><a className="text-link" href="#history">Full history</a></div><TransactionTable rows={predictions.slice(0, 5)} compact /></section>
    <p className="data-note">{busy ? 'Refreshing live metrics…' : "Dashboard values come from this app API and locally stored predictions."}</p>
  </div>
}

function TransactionTable({ rows, compact = false, onLabel }) {
  if (!rows?.length) return <div className="empty-table"><svg className="empty-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 4h8l3 3v13H5V4h3Zm0 7h8m-8 4h8" /></svg><b>No transactions yet</b><small>Once you score a transaction, its prediction will be listed here.</small></div>
  return <div className="table-wrap"><table><thead><tr><th>TRANSACTION</th><th>TYPE</th><th>AMOUNT</th><th>RISK SCORE</th><th>DECISION</th><th>MODEL</th>{!compact && <th>TRUE LABEL</th>}</tr></thead><tbody>
    {rows.map((row) => <tr key={row.transaction_id}><td><span className="transaction-id">{shortId(row.transaction_id)}</span><small className="table-sub">{new Date(row.timestamp).toLocaleString()}</small></td><td>{row.type}</td><td>${formatNumber(row.amount, 2)}</td><td><span className="risk-score">{(row.fraud_probability * 100).toFixed(1)}%</span></td><td><span className={`decision ${row.is_fraud ? 'fraud' : 'safe'}`}><i />{row.is_fraud ? 'Flagged' : 'Clear'}</span></td><td><span className="table-model">{row.model_version}</span></td>{!compact && <td>{row.true_label === null || row.true_label === undefined ? <button className="label-button" onClick={() => onLabel?.(row, row.is_fraud ? 1 : 0)}>Add label</button> : <span className={`decision ${row.true_label ? 'fraud' : 'safe'}`}><i />{row.true_label ? 'Fraud' : 'Legitimate'}</span>}</td>}</tr>)}
  </tbody></table></div>
}

function PredictPage({ onScored }) {
  const [form, setForm] = useState(EMPTY_FORM)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const update = (event) => setForm((current) => ({ ...current, [event.target.name]: event.target.value }))
  const submit = async (event) => {
    event.preventDefault(); setBusy(true); setError(''); setResult(null)
    const payload = { ...form, step: Number(form.step) }
    for (const field of ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']) payload[field] = Number(form[field])
    try { const prediction = await predictTransaction(payload); setResult(prediction); onScored?.() }
    catch (requestError) { setError(requestError.message) }
    finally { setBusy(false) }
  }
  return <div className="page-content">
    <PageTitle eyebrow="RISK ENGINE / SINGLE TRANSACTION" title={<>A clearer view of<br /><em>transaction risk.</em></>} description="Score a payment with the active production model. Input is validated before inference." />
    <div className="predict-layout">
      <section className="panel form-panel"><div className="panel-head"><div><span className="panel-kicker">TRANSACTION INPUT</span><h2>Enter transaction details</h2></div><span className="tag">7 REQUIRED FIELDS</span></div>
        <div className="preset-row"><span>QUICK SCENARIOS</span><button type="button" onClick={() => setForm(PRESETS.safe)}>Routine payment <i className="preset-safe" /></button><button type="button" onClick={() => setForm(PRESETS.elevated)}>High-value transfer <i className="preset-risk" /></button></div>
        <form onSubmit={submit}><div className="form-grid"><label className="field full"><span>Transaction type</span><select name="type" value={form.type} onChange={update}>{TYPES.map((type) => <option key={type}>{type}</option>)}</select></label>
          {[
            ['step', 'Transaction step', 'Hour index from the PaySim simulation'],
            ['amount', 'Amount', 'Value transferred'],
            ['oldbalanceOrg', 'Origin balance before', 'Sender balance before transaction'],
            ['newbalanceOrig', 'Origin balance after', 'Sender balance after transaction'],
            ['oldbalanceDest', 'Destination balance before', 'Recipient balance before transaction'],
            ['newbalanceDest', 'Destination balance after', 'Recipient balance after transaction'],
          ].map(([name, label, helper]) => <label className="field" key={name}><span>{label}</span><input name={name} value={form[name]} onChange={update} type="number" min="0" step={name === 'step' ? '1' : 'any'} required /><small>{helper}</small></label>)}
        </div>
        {error && <div className="notice notice-error" role="alert">{error}</div>}
        <button className="button button-primary score-button" disabled={busy}>{busy ? <><span className="spinner" /> Scoring transaction…</> : <>Run risk assessment <span aria-hidden="true">→</span></>}</button></form>
      </section>
      <section className={`panel assessment-panel ${result?.is_fraud ? 'is-fraud' : ''}`}><div className="panel-head"><div><span className="panel-kicker">LIVE INFERENCE</span><h2>Risk signal</h2></div><span className="tag tag-live"><i /> ONLINE</span></div>
        <RiskOrb probability={result?.fraud_probability ?? 0.08} active={busy || Boolean(result)} scoring={busy} />
        {result ? <div className="assessment-result" aria-live="polite"><span className={`decision ${result.is_fraud ? 'fraud' : 'safe'}`}><i />{result.is_fraud ? 'FRAUD · Elevated risk' : 'LEGITIMATE · Within expected range'}</span><strong><CountUp value={`${(100 * result.fraud_probability).toFixed(2)}%`} /></strong><div className="probability-bar" role="img" aria-label={`Risk probability ${(100 * result.fraud_probability).toFixed(2)} percent; decision threshold ${(100 * result.decision_threshold).toFixed(2)} percent`} style={{ '--probability': `${100 * result.fraud_probability}%`, '--threshold': `${100 * result.decision_threshold}%` }}><i /></div><div className="result-meta"><span>Threshold <b>{(100 * result.decision_threshold).toFixed(2)}%</b></span><span>Model <b>{result.model_version}</b></span><span>Latency <b>{formatNumber(result.latency_ms, 1)} ms</b></span></div></div> : <div className="orb-prompt"><strong>{busy ? 'Scoring transaction…' : 'Waiting for a transaction'}</strong><span>Probability is visualized when the model returns a score.</span></div>}
        <div className="assessment-foot"><span>Prediction ID</span><code>{shortId(result?.transaction_id)}</code></div>
      </section>
    </div>
  </div>
}

function BatchPage({ onScored }) {
  const [input, setInput] = useState('')
  const [drag, setDrag] = useState(false)
  const [results, setResults] = useState([])
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const handleFile = async (event) => {
    const file = event.target.files?.[0]
    if (file) setInput(await file.text())
  }
  const handleDrop = async (event) => { event.preventDefault(); setDrag(false); const file = event.dataTransfer.files?.[0]; if (file) setInput(await file.text()) }
  const runBatch = async () => {
    setBusy(true); setError('')
    try {
      const lines = input.trim().split(/\r?\n/).filter(Boolean)
      const headers = lines.shift().split(',').map((header) => header.trim())
      const transactions = lines.slice(0, 500).map((line) => {
        const values = line.split(',')
        const row = Object.fromEntries(headers.map((header, index) => [header, values[index]?.trim()]))
        for (const field of ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']) row[field] = Number(row[field])
        return row
      })
      if (!transactions.length) throw new Error('Add a CSV with headers and at least one transaction.')
      const data = await predictBatch(transactions); setResults(data); onScored?.()
    } catch (requestError) { setError(requestError.message) }
    finally { setBusy(false) }
  }
  return <div className="page-content"><PageTitle eyebrow="RISK ENGINE / BULK SCORING" title={<>One file. <em>Every signal.</em></>} description="Upload a transaction CSV or paste rows to score up to 500 transactions in a single request." />
    <div className="batch-layout"><section className={`panel batch-input ${drag ? 'is-dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDrag(true) }} onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) setDrag(false) }} onDrop={handleDrop}><div className="panel-head"><div><span className="panel-kicker">BATCH SOURCE</span><h2>Import transactions</h2></div><span className="tag">CSV · MAX 500 ROWS</span></div><label className="upload-zone"><input type="file" accept=".csv,text/csv" onChange={handleFile} /><span className="upload-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M12 16V4m-5 5 5-5 5 5M4 16v4h16v-4" /></svg></span><b>Drop a CSV here or browse files</b><small>Required headers: step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest</small></label><textarea value={input} onChange={(event) => setInput(event.target.value)} placeholder={'step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest\\n120,TRANSFER,50000,50000,0,0,0'} aria-label="Transaction CSV contents" />{error && <div className="notice notice-error" role="alert">{error}</div>}{busy && <div className="batch-progress" role="status">Scoring batch<span /></div>}<button className="button button-primary" onClick={runBatch} disabled={busy || !input.trim()}>{busy ? 'Scoring batch…' : 'Score batch'}</button></section>
      <section className="panel batch-output"><div className="panel-head"><div><span className="panel-kicker">BATCH RESULTS</span><h2>{results.length ? `${results.length} transactions scored` : 'Results preview'}</h2></div></div>{results.length ? <TransactionTable rows={results} compact /> : <div className="empty-table"><svg className="empty-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 4h8l3 3v13H5V4h3Zm0 7h8m-8 4h8" /></svg><b>Ready for your file</b><small>Validated scores and model decisions will appear here.</small></div>}</section></div>
  </div>
}

function MonitoringPage({ data, error }) {
  const summary = data.summary || {}
  const drift = data.drift || {}
  const rows = Object.entries(drift.features || {})
  const recent = data.predictions?.items || []
  const riskPoints = recent.slice(0, 18)
  const distribution = Array.from({ length: 12 }, (_, i) => recent.filter((row) => Math.min(11, Math.floor(row.fraud_probability * 12)) === i).length)
  return <div className="page-content"><PageTitle eyebrow="OBSERVABILITY / MODEL MONITORING" title={<>Signals, shifts, <em>stability.</em></>} description="Observe API health, prediction behavior, labeled outcomes and input drift against the production reference." />
    {error && <div className="notice notice-error" role="alert">{error}</div>}
    <section className="stat-grid monitoring-stats"><StatCard loading={!data.health && !error} label="API requests" value={formatNumber(summary.api_requests)} note="Since current process started" icon="transactions" /><StatCard loading={!data.health && !error} label="P95 latency" value={`${formatNumber(summary.latency_p95_ms, 1)} ms`} note={`P50 ${formatNumber(summary.latency_p50_ms, 1)} ms`} icon="transactions" tone="blue" /><StatCard loading={!data.health && !error} label="API error rate" value={`${(100 * (summary.error_rate || 0)).toFixed(2)}%`} note={`${formatNumber(summary.api_errors)} server errors`} icon="fraud" tone="red" /><StatCard loading={!data.health && !error} label="Labeled outcomes" value={formatNumber(summary.performance?.labeled_predictions)} note="Ground truth attached later" icon="transactions" tone="violet" /></section>
    <section className="overview-grid monitoring-grid"><article className="panel"><div className="panel-head"><div><span className="panel-kicker">PREDICTION DRIFT</span><h2>Recent prediction scatter</h2></div><span className="tag">RECENT</span></div><div className="distribution-chart">{distribution.map((count, index) => <div key={index} className="distribution-column"><i style={{ height: `${count ? Math.max(6, count / Math.max(...distribution, 1) * 100) : 2}%` }} /><small>{Math.round(index * 100 / 12)}%</small></div>)}</div><PredictionScatter predictions={riskPoints} /><div className="chart-footer"><span>Recent risk distribution ? hover points for details</span><b>{recent.length} samples</b></div></article>
      <article className="panel"><div className="panel-head"><div><span className="panel-kicker">DATA DRIFT · PSI</span><h2>Feature stability</h2></div><span className={`drift-state ${drift.status || 'stable'}`}><i />{String(drift.status || 'waiting').replace('_', ' ')}</span></div>{rows.length ? <div className="drift-list">{rows.map(([name, value]) => <div className="drift-row" key={name}><span>{name}</span><div className="drift-track"><i className={value.status} style={{ width: `${Math.min(100, value.psi / 0.3 * 100)}%` }} /></div><b>{value.psi.toFixed(3)}</b><small>{value.status}</small></div>)}</div> : <div className="empty-inline">{drift.sample_size ? 'No comparable input features in reference.' : 'Drift metrics appear after scored transactions are available.'}</div>}<div className="chart-footer"><span>Warning 0.10 · alert 0.25 PSI</span><span>KS statistic included</span></div></article></section>
    <section className="panel monitoring-bottom"><div className="panel-head"><div><span className="panel-kicker">LABELED PERFORMANCE</span><h2>Feedback loop</h2></div><a className="text-link" href="#history">Attach labels in history</a></div><div className="feedback-metrics"><div><span>Precision</span><b>{summary.performance?.precision == null ? 'Waiting for labels' : `${(100 * summary.performance.precision).toFixed(1)}%`}</b></div><div><span>Recall</span><b>{summary.performance?.recall == null ? 'Waiting for labels' : `${(100 * summary.performance.recall).toFixed(1)}%`}</b></div><div><span>PR-AUC</span><b>{summary.performance?.pr_auc == null ? 'Waiting for both classes' : summary.performance.pr_auc.toFixed(4)}</b></div></div><small className="scope-note">Operational request metrics are process-local and reset on restart; transaction history is persisted in JSONL.</small></section>
  </div>
}

function ModelPage({ data, error }) {
  const model = data.model || {}
  const metrics = model.metrics || {}
  const importance = Object.entries(metrics.feature_importance || {}).sort((a, b) => b[1] - a[1]).slice(0, 8)
  const cm = model.confusion_matrix || {}
  const matrix = [['TN', cm.tn], ['FP', cm.fp], ['FN', cm.fn], ['TP', cm.tp]]
  const max = Math.max(...matrix.map(([, value]) => Number(value) || 0), 1)
  return <div className="page-content"><PageTitle eyebrow="MODEL OPS / REGISTRY" title={<>The model behind <em>the signal.</em></>} description="Production artifact, decision policy, held-out evaluation and feature contribution at a glance." />{error && <div className="notice notice-error" role="alert">{error}</div>}
    <section className="model-hero panel"><div className="model-hex"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 3 9 5v8l-9 5-9-5V8l9-5Zm-9 5 9 5 9-5m-9 5v8" /></svg><i /><i /><i /></div><div className="model-hero-copy"><span className="tag tag-live"><i /> PRODUCTION</span><h2>{model.model || 'Model unavailable'} <small>{model.version || '—'}</small></h2><p>PaySim fraud classifier · probability-based decisioning</p><div className="model-hero-meta"><span>Threshold <b>{model.threshold == null ? '—' : `${(100 * model.threshold).toFixed(2)}%`}</b></span><span>Size <b>{model.model_size_bytes ? `${(model.model_size_bytes / 1024).toFixed(0)} KB` : '—'}</b></span><span>Trained <b>{model.training_date ? new Date(model.training_date).toLocaleDateString() : '—'}</b></span></div></div></section>
    <section className="model-metric-grid">{[['PR-AUC', metrics.pr_auc?.toFixed?.(4)], ['ROC-AUC', metrics.roc_auc?.toFixed?.(4)], ['Precision', metrics.precision == null ? null : `${(100 * metrics.precision).toFixed(2)}%`], ['Recall', metrics.recall == null ? null : `${(100 * metrics.recall).toFixed(2)}%`], ['F2 score', metrics.f2?.toFixed?.(4)]].map(([label, value]) => <div className="panel compact-metric" key={label}><span>{label}</span><b>{value ?? '—'}</b></div>)}</section>
    <section className="overview-grid"><article className="panel"><div className="panel-head"><div><span className="panel-kicker">HELD-OUT TEST</span><h2>Confusion matrix</h2></div><span className="tag">TEST SPLIT</span></div><ConfusionBars values={matrix} /><div className="matrix-bars">{matrix.map(([label, value]) => <div className="matrix-bar" key={label}><div className="matrix-bar-top"><span>{label}</span><b>{formatNumber(value)}</b></div><div className="matrix-track"><i className={label} style={{ width: `${Math.max(1, Math.log1p(value || 0) / Math.log1p(max) * 100)}%` }} /></div></div>)}</div><p className="chart-footnote">Bar scale is logarithmic to keep minority error counts visible.</p></article>
      <article className="panel"><div className="panel-head"><div><span className="panel-kicker">XGBOOST BUILT-IN</span><h2>Feature importance</h2></div><span className="tag">TOP FEATURES</span></div>{importance.length ? <div className="importance-list">{importance.map(([name, score]) => <div className="importance-row" key={name}><span>{name}</span><div className="importance-track"><i style={{ width: `${score / Math.max(...importance.map(([, value]) => value), 1e-9) * 100}%` }} /></div><b>{score.toFixed(3)}</b></div>)}</div> : <div className="empty-inline">Feature importance unavailable for this model.</div>}</article></section>
    <div className="caveat-note"><span aria-hidden="true">→</span><p><b>Evaluation context</b><br />Metrics reflect a synthetic PaySim dataset and a time-ordered holdout. They should not be interpreted as expected real-world fraud performance.</p></div>
  </div>
}

function HistoryPage({ data, onRefresh, error }) {
  const [filter, setFilter] = useState('all')
  const [labelError, setLabelError] = useState('')
  const rows = data.predictions?.items || []
  const filtered = filter === 'all' ? rows : rows.filter((row) => filter === 'fraud' ? row.is_fraud : !row.is_fraud)
  const attach = async (row, label) => {
    setLabelError('')
    try { await patchLabel(row.transaction_id, label); onRefresh() } catch (requestError) { setLabelError(requestError.message) }
  }
  return <div className="page-content"><PageTitle eyebrow="TRANSACTION LEDGER / PREDICTION HISTORY" title={<>Every decision, <em>accounted for.</em></>} description="Review recent scores, filter model decisions and attach known labels for performance monitoring." action={<button className="button button-secondary" onClick={onRefresh}><span aria-hidden="true">→</span> Refresh</button>} />
    {error && <div className="notice notice-error" role="alert">{error}</div>}{labelError && <div className="notice notice-error" role="alert">{labelError}</div>}<section className="panel history-panel"><div className="panel-head"><div><span className="panel-kicker">PERSISTED PREDICTIONS</span><h2>{formatNumber(data.predictions?.total)} transactions</h2></div><div className="filter-tabs">{[['all', 'All'], ['fraud', 'Flagged'], ['clear', 'Clear']].map(([value, label]) => <button className={filter === value ? 'selected' : ''} onClick={() => setFilter(value)} key={value}>{label}</button>)}</div></div><TransactionTable rows={filtered} onLabel={attach} /></section><p className="data-note">Adding an actual label updates model performance metrics on the Monitoring page.</p>
  </div>
}

export default function App() {
  const page = usePage()
  const [theme, setTheme] = useState(() => { try { return localStorage.getItem('sentinel-theme') === 'light' ? 'light' : 'dark' } catch { return 'dark' } })
  const [reducedMotion, setReducedMotion] = useState(() => { try { return localStorage.getItem('sentinel-reduced-motion') === 'true' || window.matchMedia('(prefers-reduced-motion: reduce)').matches } catch { return false } })
  useEffect(() => { document.documentElement.dataset.theme = theme; try { localStorage.setItem('sentinel-theme', theme) } catch {} }, [theme])
  useEffect(() => { document.documentElement.dataset.reduceMotion = String(reducedMotion); try { localStorage.setItem('sentinel-reduced-motion', String(reducedMotion)) } catch {} }, [reducedMotion])
  const [data, setData] = useState({ summary: {}, predictions: { items: [], total: 0 }, drift: {}, model: {}, health: null })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const refresh = useCallback(async () => {
    setBusy(true); setError('')
    try { setData(await getDashboardData()) } catch (requestError) { setError(requestError.message) }
    finally { setBusy(false) }
  }, [])
  useEffect(() => { refresh(); const interval = window.setInterval(refresh, 30000); return () => window.clearInterval(interval) }, [refresh])
  const pageTitle = NAV_ITEMS.find((item) => item.id === page)?.label || 'Overview'
  let content
  if (page === 'predict') content = <PredictPage onScored={refresh} />
  else if (page === 'batch') content = <BatchPage onScored={refresh} />
  else if (page === 'monitoring') content = <MonitoringPage data={data} error={error} />
  else if (page === 'model') content = <ModelPage data={data} error={error} />
  else if (page === 'history') content = <HistoryPage data={data} onRefresh={refresh} error={error} />
  else content = <OverviewPage data={data} busy={busy} error={error} />
  return <div className="app-shell"><Sidebar page={page} /><main className="main-area"><Topbar title={pageTitle} health={data.health} theme={theme} setTheme={setTheme} reducedMotion={reducedMotion} setReducedMotion={setReducedMotion} />{content}<footer className="app-footer"><span>FRAUD DETECTION ? FRAUD INTELLIGENCE</span><span>FASTAPI <i /> REACT <i /> THREE.JS</span></footer></main></div>
}
