import { useEffect, useState } from 'react';

type Candidate = { item_id: number; relevant: boolean | null; filename: string; document_id: number; page_number: number; content: string; preview_url: string | null };
type Query = { id: number; query: string; modality: string; verified: boolean; candidates: Candidate[] };
type Readiness = { sources_missing: number; licenses_missing: number; figures: number; verified_text_queries: number; verified_image_queries: number; active_learning: Record<string, { labels: number; ready: boolean; blocking_reason: string | null; classes: Record<string, number> }> };
type Job = { id: string; status: string; output: string; experiment: string };
async function request<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch('/api/v1/research' + path, body === undefined ? undefined : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const data = await r.json(); if (!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Request failed'); return data;
}
export default function ResearchPanel() {
  const [state, setState] = useState<Readiness | null>(null);
  const [queries, setQueries] = useState<Query[]>([]);
  const [query, setQuery] = useState('');
  const [modality, setModality] = useState('text');
  const [selected, setSelected] = useState<number | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [itemType, setItemType] = useState('passage');
  async function refresh() { const [s,q] = await Promise.all([request<Readiness>('/readiness'), request<Query[]>('/queries')]); setState(s); setQueries(q); }
  useEffect(() => { refresh().catch(e => setError(e.message)); }, []);
  useEffect(() => {
    if (!job || !['queued','running'].includes(job.status)) return;
    const timer = window.setInterval(() => { request<Job>(`/runs/${job.id}`).then(j => { setJob(j); if (['completed','failed'].includes(j.status)) refresh().catch(e => setError(e.message)); }).catch(e => { setError(e.message); setJob(null); }); }, 2000);
    return () => window.clearInterval(timer);
  }, [job?.id, job?.status]);
  async function act(operation: () => Promise<unknown>) { setBusy(true); setError(''); try { await operation(); await refresh(); } catch(e) { setError((e as Error).message); } finally { setBusy(false); } }
  const active = queries.find(q => q.id === selected);
  const running = !!job && ['queued','running'].includes(job.status);
  return <section className="research-panel">
    <div className="section-heading"><h2>Finish your research workflow</h2><button className="text-button" onClick={() => act(refresh)}>Check readiness</button></div>
    {error && <div className="alert error" role="alert">{error}</div>}
    {state && <div className="readiness-grid"><div><strong>1. Restore sources</strong><p>{state.sources_missing} source files and {state.licenses_missing} retained-source licenses missing. {state.figures} figures indexed.</p><p>Attach originals in Library, then record the actual reuse license.</p></div><div><strong>2. Add human labels</strong>{Object.entries(state.active_learning).map(([kind,s]) => <p key={kind}><b>{kind}: {s.labels} labels</b> · {s.ready ? 'Ready to run' : s.blocking_reason}</p>)}<p>Go to Review. Use the real class; do not force class balance by assigning incorrect labels.</p><a className="text-button" href="/api/v1/research/labels/export" target="_blank" rel="noreferrer">Export human labels ↗</a></div><div><strong>3. Judge retrieval relevance</strong><p>{state.verified_text_queries} text queries and {state.verified_image_queries} image queries verified.</p><p>Create queries below, pool results, judge every candidate, then verify. These judgments are separate from anomaly labels.</p></div></div>}
    <details className="judging-workspace"><summary>Human relevance judgments</summary><p className="helper">Pool the top 5 results from every applicable method. Judge relevance to the query, independent of retrieval rank. Recall is relative to this pool; more exhaustive judgments give stronger evidence.</p>
      <form className="query-form" onSubmit={e => { e.preventDefault(); act(async () => { const q = await request<Query>('/queries',{query,modality}); setSelected(q.id); setQuery(''); }); }}><input aria-label="Evaluation query" required minLength={3} maxLength={500} placeholder="Write a held-out evaluation question…" value={query} onChange={e => setQuery(e.target.value)}/><select aria-label="Evaluation modality" value={modality} onChange={e => setModality(e.target.value)}><option value="text">Text</option><option value="image">Image</option></select><button className="button primary" disabled={busy}>Add query</button></form>
      <div className="query-choices">{queries.map(q => <button className={`button ${selected === q.id ? 'primary' : 'secondary'}`} key={q.id} onClick={() => setSelected(q.id)}>{q.modality} #{q.id} {q.verified ? '✓' : '· draft'}</button>)}</div>
      {active && <div className="judge-query"><h3>{active.query}</h3><div className="query-actions"><button className="button secondary" disabled={busy || running} onClick={() => act(() => request(`/queries/${active.id}/pool`,{}))}>Pool search results</button><button className="button primary" disabled={busy || active.verified || !active.candidates.length || active.candidates.some(c => c.relevant === null)} onClick={() => act(() => request(`/queries/${active.id}/verify`,{}))}>{active.verified ? 'Verified by human' : 'Verify completed judgments'}</button><span>{active.candidates.filter(c => c.relevant !== null).length}/{active.candidates.length} judged</span></div>
        {active.candidates.map(c => <article className="judge-candidate" key={c.item_id}><small>{c.filename} · page {c.page_number} · item #{c.item_id}</small>{c.preview_url && <img src={c.preview_url} alt="Figure to judge"/>}<p>{c.content || 'Inspect the figure for relevance.'}</p><div><button aria-pressed={c.relevant === true} className={`button ${c.relevant === true ? 'primary' : 'secondary'}`} disabled={busy} onClick={() => act(() => request(`/queries/${active.id}/judgments`,{item_id:c.item_id,relevant:true}))}>Relevant</button><button aria-pressed={c.relevant === false} className={`button ${c.relevant === false ? 'primary' : 'secondary'}`} disabled={busy} onClick={() => act(() => request(`/queries/${active.id}/judgments`,{item_id:c.item_id,relevant:false}))}>Not relevant</button><a href={`/api/v1/documents/${c.document_id}/source#page=${c.page_number}`} target="_blank" rel="noreferrer">Open source ↗</a></div></article>)}
      </div>}<a className="text-button" href="/api/v1/research/queries/export" target="_blank" rel="noreferrer">Export verified query set ↗</a>
    </details>
    <div className="experiment-controls"><h3>Run experiments</h3><p className="helper">Runs continue in the background. Keep the backend running; run tracking resets on restart. Refresh results below after completion.</p><div className="query-actions"><button className="button secondary" disabled={busy || running || !state || state.verified_text_queries + state.verified_image_queries === 0} onClick={() => act(async () => setJob(await request<Job>('/runs',{experiment:'retrieval'})))}>Evaluate judged queries</button><select aria-label="Active learning modality" value={itemType} onChange={e => setItemType(e.target.value)}><option value="passage">Passage labels</option><option value="figure">Figure labels</option></select><button className="button secondary" disabled={busy || running || !state?.active_learning[itemType].ready} onClick={() => act(async () => setJob(await request<Job>('/runs',{experiment:'active_learning',item_type:itemType})))}>Compare active learning</button><button className="button secondary" disabled={busy || running} onClick={() => act(async () => setJob(await request<Job>('/runs',{experiment:'index'})))}>Benchmark index</button></div>{job && <div role="status" className="job-status"><strong>{job.experiment}: {job.status}</strong>{job.output && <details><summary>Run output</summary><pre>{job.output}</pre></details>}</div>}</div>
  </section>;
}
