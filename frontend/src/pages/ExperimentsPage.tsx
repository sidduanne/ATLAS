import { useEffect, useState } from 'react';
import { BarChart3, FlaskConical } from 'lucide-react';
import PageHeading from '../components/PageHeading';
import ResearchPanel from '../components/ResearchPanel';
import { api, message, type Analytics, type Notify } from '../api';

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
