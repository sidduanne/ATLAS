import { useEffect, useState } from 'react';
import { BarChart3, FlaskConical } from 'lucide-react';
import { PageHeading } from './ui';
import { api, json, message, sourceUrl, type Analytics, type Notify } from './api';

type Reports = Record<string, Record<string, unknown>>;

const EXPERIMENTS = [
  { key: 'retrieval_summary', title: '01 / Retrieval effectiveness', script: 'evaluate_retrieval.py',
    description: 'Keyword, Sentence-BERT, hybrid, and CLIP retrieval. Precision@5, Recall@5, MRR, and NDCG@5.' },
  { key: 'active_learning', title: '02 / Label efficiency', script: 'evaluate_active_learning.py',
    description: 'Entropy and margin versus random selection under equal budgets, using a fixed held-out set.' },
  { key: 'index_benchmark', title: '03 / Index performance', script: 'benchmark_index.py',
    description: 'Exact versus HNSW at increasing corpus sizes. Median, p95 latency, and Recall@K.' },
];

export default function ExperimentsPage({ stats, refresh, notify }: { stats: Analytics | null; refresh: () => Promise<void>; notify: Notify }) {
  const [reports, setReports] = useState<Reports>({});
  const load = () => api<{ reports: Reports }>('/experiments').then(r => setReports(r.reports)).catch(e => notify.error(message(e)));
  useEffect(() => { load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const latency = [['Median search latency', stats?.search_latency.median_ms], ['95th percentile', stats?.search_latency.p95_ms]] as const;
  return (
    <>
      <PageHeading eyebrow="EVIDENCE OVER ASSUMPTIONS" title="Measure what matters."
        subtitle="Retrieval quality, label efficiency, and the speed–recall trade-off.">
        <button className="button secondary" onClick={() => { load(); refresh().catch(e => notify.error(message(e))); }}>Refresh results</button>
      </PageHeading>

      <div className="metrics-grid">
        {latency.map(([name, value]) => (
          <div className="metric-card" key={name}>
            <BarChart3 size={20} />
            <span>{name}</span>
            <strong>{value == null ? '—' : `${value.toFixed(1)} ms`}</strong>
            <small>{stats?.search_latency.sample_count || 0} successful requests · current server process</small>
          </div>
        ))}
      </div>

      <ResearchPanel />

      <div className="experiment-list">
        {EXPERIMENTS.map(e => (
          <section className="experiment-card" key={e.key}>
            <div><h2>{e.title}</h2><span className="status-pill">{reports[e.key] ? 'Report available' : 'Awaiting experiment'}</span></div>
            <p>{e.description}</p>
            <Report data={reports[e.key]} />
            <details>
              <summary>Run this experiment</summary>
              <p className="helper">Run from the repository root while the stack is running. See README for labels, judgments, and options.</p>
              <code>docker compose exec backend python experiments/{e.script}</code>
            </details>
          </section>
        ))}
      </div>
      <p className="helper">Experiments use real corpus data. Verify provisional judgments before reporting conclusions. Missing experiments have no generated scores.</p>
    </>
  );
}

type Row = Record<string, number | string>;
const fixed = (value: unknown, digits = 3) => (typeof value === 'number' ? value.toFixed(digits) : '—');

function Table({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <table className="metrics-table">
      <thead><tr>{head.map(h => <th key={h}>{h}</th>)}</tr></thead>
      <tbody>{rows.map((row, i) => <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>)}</tbody>
    </table>
  );
}

/** Render a saved report: retrieval metrics, learning curves, index benchmark, or raw JSON. */
function Report({ data }: { data?: Record<string, unknown> }) {
  if (!data) return <div className="report-empty"><FlaskConical size={18} /> Run the experiment to see measured results here.</div>;
  if (data.macro_averages) {
    const averages = Object.entries(data.macro_averages as Record<string, Row>);
    return <>
      <div className="report-note">
        {data.provisional_relevance_judgments ? 'PROVISIONAL JUDGMENTS · ' : ''}{String(data.query_count)} queries · {String(data.failure_count || 0)} failures
      </div>
      <Table head={['Method', 'P@5', 'R@5', 'MRR', 'NDCG@5']}
        rows={averages.map(([name, v]) => [name, fixed(v.precision_at_5), fixed(v.recall_at_5), fixed(v.mrr), fixed(v.ndcg_at_5)])} />
    </>;
  }
  if (data.curves) {
    return <>
      <div className="report-note">HELD-OUT MACRO F1 · {String(data.heldout_count)} items · {String(data.seeds)} seeds</div>
      <Table head={['Selection', 'Labels', 'Macro F1', 'Std.']}
        rows={(data.curves as Row[]).map(v => [v.strategy, v.labels_reviewed, fixed(v.mean_f1), fixed(v.std_f1)])} />
    </>;
  }
  if (data.measurements) {
    return <Table head={['Corpus', 'Index', 'Median', 'p95', 'Recall@K']}
      rows={(data.measurements as Row[]).map(v => [v.corpus_size, v.mode, `${fixed(v.median_ms, 2)} ms`, `${fixed(v.p95_ms, 2)} ms`, fixed(v.recall_at_k)])} />;
  }
  return <pre className="raw-report">{JSON.stringify(data, null, 2)}</pre>;
}

// --- Research workflow: readiness, relevance judging, experiment runs ---------------------------

type Candidate = { item_id: number; relevant: boolean | null; filename: string; document_id: number; page_number: number; content: string; preview_url: string | null };
type Query = { id: number; query: string; modality: string; verified: boolean; candidates: Candidate[] };
type Readiness = {
  sources_missing: number; licenses_missing: number; figures: number; verified_text_queries: number; verified_image_queries: number;
  active_learning: Record<string, { labels: number; ready: boolean; blocking_reason: string | null }>;
};
type Job = { id: string; status: string; output: string; experiment: string };

const research = <T,>(path: string, body?: unknown) => api<T>(`/research${path}`, body === undefined ? undefined : json(body));
const isRunning = (job: Job | null) => !!job && ['queued', 'running'].includes(job.status);

/** Readiness checklist, human relevance judging, and background experiment runs. */
function ResearchPanel() {
  const [state, setState] = useState<Readiness | null>(null);
  const [queries, setQueries] = useState<Query[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function refresh() {
    const [readiness, list] = await Promise.all([research<Readiness>('/readiness'), research<Query[]>('/queries')]);
    setState(readiness);
    setQueries(list);
  }
  useEffect(() => { refresh().catch(e => setError(message(e))); }, []);

  // Poll the running experiment; refresh readiness once it finishes.
  useEffect(() => {
    if (!job || !isRunning(job)) return;
    const timer = window.setInterval(() => {
      research<Job>(`/runs/${job.id}`)
        .then(next => { setJob(next); if (!isRunning(next)) refresh().catch(e => setError(message(e))); })
        .catch(e => { setError(message(e)); setJob(null); });
    }, 2000);
    return () => window.clearInterval(timer);
  }, [job?.id, job?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  async function act(operation: () => Promise<unknown>) {
    setBusy(true);
    setError('');
    try { await operation(); await refresh(); } catch (e) { setError(message(e)); } finally { setBusy(false); }
  }
  const run = (body: object) => act(async () => setJob(await research<Job>('/runs', body)));
  const active = queries.find(q => q.id === selected);
  const running = isRunning(job);

  return (
    <section className="research-panel">
      <div className="section-heading"><h2>Finish your research workflow</h2><button className="text-button" onClick={() => act(refresh)}>Check readiness</button></div>
      {error && <div className="alert error" role="alert">{error}</div>}
      {state && <Readiness state={state} />}

      <details className="judging-workspace">
        <summary>Human relevance judgments</summary>
        <p className="helper">Pool the top 5 results from every applicable method. Judge relevance to the query, independent of retrieval rank. Recall is relative to this pool; more exhaustive judgments give stronger evidence.</p>
        <NewQueryForm busy={busy} onCreate={(query, modality, clear) => act(async () => {
          setSelected((await research<Query>('/queries', { query, modality })).id);
          clear();
        })} />
        <div className="query-choices">
          {queries.map(q => (
            <button key={q.id} className={`button ${selected === q.id ? 'primary' : 'secondary'}`} onClick={() => setSelected(q.id)}>
              {q.modality} #{q.id} {q.verified ? '✓' : '· draft'}
            </button>
          ))}
        </div>
        {active && <JudgeQuery query={active} busy={busy} running={running} act={act} />}
        <a className="text-button" href="/api/v1/research/queries/export" target="_blank" rel="noreferrer">Export verified query set ↗</a>
      </details>

      <ExperimentControls state={state} busy={busy} running={running} job={job} run={run} />
    </section>
  );
}

function Readiness({ state }: { state: Readiness }) {
  return (
    <div className="readiness-grid">
      <div>
        <strong>1. Restore sources</strong>
        <p>{state.sources_missing} source files and {state.licenses_missing} retained-source licenses missing. {state.figures} figures indexed.</p>
        <p>Attach originals in Library, then record the actual reuse license.</p>
      </div>
      <div>
        <strong>2. Add human labels</strong>
        {Object.entries(state.active_learning).map(([kind, s]) => (
          <p key={kind}><b>{kind}: {s.labels} labels</b> · {s.ready ? 'Ready to run' : s.blocking_reason}</p>
        ))}
        <p>Go to Review. Use the real class; do not force class balance by assigning incorrect labels.</p>
        <a className="text-button" href="/api/v1/research/labels/export" target="_blank" rel="noreferrer">Export human labels ↗</a>
      </div>
      <div>
        <strong>3. Judge retrieval relevance</strong>
        <p>{state.verified_text_queries} text queries and {state.verified_image_queries} image queries verified.</p>
        <p>Create queries below, pool results, judge every candidate, then verify. These judgments are separate from anomaly labels.</p>
      </div>
    </div>
  );
}

function NewQueryForm({ busy, onCreate }: { busy: boolean; onCreate: (query: string, modality: string, clear: () => void) => void }) {
  const [query, setQuery] = useState('');
  const [modality, setModality] = useState('text');
  return (
    <form className="query-form" onSubmit={e => { e.preventDefault(); onCreate(query, modality, () => setQuery('')); }}>
      <input aria-label="Evaluation query" required minLength={3} maxLength={500} placeholder="Write a held-out evaluation question…"
        value={query} onChange={e => setQuery(e.target.value)} />
      <select aria-label="Evaluation modality" value={modality} onChange={e => setModality(e.target.value)}>
        <option value="text">Text</option><option value="image">Image</option>
      </select>
      <button className="button primary" disabled={busy}>Add query</button>
    </form>
  );
}

function JudgeQuery({ query, busy, running, act }: { query: Query; busy: boolean; running: boolean; act: (op: () => Promise<unknown>) => Promise<void> }) {
  const judged = query.candidates.filter(c => c.relevant !== null).length;
  const judge = (itemId: number, relevant: boolean) => act(() => research(`/queries/${query.id}/judgments`, { item_id: itemId, relevant }));
  return (
    <div className="judge-query">
      <h3>{query.query}</h3>
      <div className="query-actions">
        <button className="button secondary" disabled={busy || running} onClick={() => act(() => research(`/queries/${query.id}/pool`, {}))}>Pool search results</button>
        <button className="button primary" disabled={busy || query.verified || !query.candidates.length || judged < query.candidates.length}
          onClick={() => act(() => research(`/queries/${query.id}/verify`, {}))}>
          {query.verified ? 'Verified by human' : 'Verify completed judgments'}
        </button>
        <span>{judged}/{query.candidates.length} judged</span>
      </div>
      {query.candidates.map(c => (
        <article className="judge-candidate" key={c.item_id}>
          <small>{c.filename} · page {c.page_number} · item #{c.item_id}</small>
          {c.preview_url && <img src={c.preview_url} alt="Figure to judge" />}
          <p>{c.content || 'Inspect the figure for relevance.'}</p>
          <div>
            {([true, false] as const).map(value => (
              <button key={String(value)} aria-pressed={c.relevant === value} disabled={busy}
                className={`button ${c.relevant === value ? 'primary' : 'secondary'}`} onClick={() => judge(c.item_id, value)}>
                {value ? 'Relevant' : 'Not relevant'}
              </button>
            ))}
            <a href={sourceUrl(c.document_id, c.page_number)} target="_blank" rel="noreferrer">Open source ↗</a>
          </div>
        </article>
      ))}
    </div>
  );
}

function ExperimentControls({ state, busy, running, job, run }:
  { state: Readiness | null; busy: boolean; running: boolean; job: Job | null; run: (body: object) => void }) {
  const [itemType, setItemType] = useState('passage');
  const blocked = busy || running;
  return (
    <div className="experiment-controls">
      <h3>Run experiments</h3>
      <p className="helper">Runs continue in the background. Keep the backend running; run tracking resets on restart. Refresh results below after completion.</p>
      <div className="query-actions">
        <button className="button secondary" disabled={blocked || !state || state.verified_text_queries + state.verified_image_queries === 0}
          onClick={() => run({ experiment: 'retrieval' })}>Evaluate judged queries</button>
        <select aria-label="Active learning modality" value={itemType} onChange={e => setItemType(e.target.value)}>
          <option value="passage">Passage labels</option><option value="figure">Figure labels</option>
        </select>
        <button className="button secondary" disabled={blocked || !state?.active_learning[itemType].ready}
          onClick={() => run({ experiment: 'active_learning', item_type: itemType })}>Compare active learning</button>
        <button className="button secondary" disabled={blocked} onClick={() => run({ experiment: 'index' })}>Benchmark index</button>
      </div>
      {job && (
        <div role="status" className="job-status">
          <strong>{job.experiment}: {job.status}</strong>
          {job.output && <details><summary>Run output</summary><pre>{job.output}</pre></details>}
        </div>
      )}
    </div>
  );
}
