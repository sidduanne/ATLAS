import { useCallback, useEffect, useRef, useState, type FormEvent } from 'react';
import { ArrowUpRight, BookOpen, Check, ChevronRight, FileText, FlaskConical, Image as ImageIcon, Layers, LoaderCircle, MessageSquare, Plus, Search, ShieldCheck, Upload, X, BarChart3 } from 'lucide-react';
import './style.css';
import ResearchPanel from './ResearchPanel';
import ImageTextDialog from './ImageTextDialog';
import ProvenanceEditor from './ProvenanceEditor';

type Tab = 'Search' | 'Library' | 'Review' | 'Experiments';
type Method = 'hybrid' | 'semantic' | 'keyword' | 'images';
type ItemType = 'passage' | 'figure';
type Result = { passage_id?: number; figure_id?: number; item_id?: number; document_id: number; filename: string; content: string; page_number: number | null; preview_url?: string; uncertainty?: number | null; prediction?: string };
type Doc = { id: number; filename: string; title: string; status: string; has_source: boolean; source_url: string | null; license: string | null };
type Analytics = { counts: Record<string, number>; search_latency: { median_ms: number | null; p95_ms: number | null; sample_count: number } };
type Queue = { results: Result[]; classifier_ready: boolean; strategy: string; labeled_count: number; remaining: number };
type Review = { item_type: ItemType; item_id: number; label: string; note: string };
type Reports = { reports: Record<string, unknown>; note: string };
const labels = ['relevant', 'duplicate', 'malformed', 'out_of_domain', 'irrelevant'];
const methodNames: Record<Method, string> = { hybrid: 'Hybrid search', semantic: 'Semantic text', keyword: 'Keyword baseline', images: 'Figures · CLIP' };
async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1${path}`, options);
  if (!response.ok) { const e = await response.json().catch(() => ({})); throw new Error(typeof e.detail === 'string' ? e.detail : `Request failed (${response.status})`); }
  return response.json() as Promise<T>;
}
const post = (body: unknown): RequestInit => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
const pretty = (s: string) => s.replaceAll('_', ' ');

export default function App() {
  const [ocrTarget, setOcrTarget] = useState<{documentId:number;filename:string;figureId?:number} | null>(null);
  const [metadataDoc, setMetadataDoc] = useState<Doc | null>(null);
  const [tab, setTab] = useState<Tab>('Search');
  const [stats, setStats] = useState<Analytics | null>(null);
  const [docs, setDocs] = useState<Doc[]>([]);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [query, setQuery] = useState('');
  const [method, setMethod] = useState<Method>('hybrid');
  const [results, setResults] = useState<Result[]>([]);
  const [searched, setSearched] = useState(false);
  const [searching, setSearching] = useState(false);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [history, setHistory] = useState<string[]>([]);
  const [turns, setTurns] = useState<{ query: string; message: string }[]>([]);
  const [conversation, setConversation] = useState(false);
  const [preview, setPreview] = useState<Result | null>(null);
  const [queue, setQueue] = useState<Queue | null>(null);
  const [strategy, setStrategy] = useState('entropy');
  const [itemType, setItemType] = useState<ItemType>('passage');
  const [skipped, setSkipped] = useState<Record<ItemType, number[]>>({passage:[],figure:[]});
  const [note, setNote] = useState('');
  const [reviewBusy, setReviewBusy] = useState(false);
  const [recent, setRecent] = useState<Review[]>([]);
  const [reports, setReports] = useState<Reports | null>(null);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [textTitle, setTextTitle] = useState('');
  const [textContent, setTextContent] = useState('');
  const [sourceUrl, setSourceUrl] = useState('');
  const [license, setLicense] = useState('');
  const [docFilter, setDocFilter] = useState('');
  const [uploadMode, setUploadMode] = useState<'files' | 'text'>('files');
  const fileRef = useRef<HTMLInputElement>(null);
  const queueRequest = useRef(0);
  const refresh = useCallback(async () => { const [s, d] = await Promise.all([api<Analytics>('/analytics'), api<Doc[]>('/documents')]); setStats(s); setDocs(d); }, []);
  useEffect(() => { refresh().catch(e => setError(e.message)); }, [refresh]);
  const loadQueue = useCallback(async () => { const request = ++queueRequest.current; const [q, r] = await Promise.all([api<Queue>(`/review/queue?item_type=${itemType}&strategy=${strategy}${skipped[itemType].map(id => `&exclude=${id}`).join('')}`), api<Review[]>('/review/labels?limit=15')]); if (request === queueRequest.current) { setQueue(q); setRecent(r); } }, [itemType, strategy, skipped]);
  useEffect(() => { if (tab === 'Review') { setQueue(null); loadQueue().catch(e => setError(e.message)); } if (tab === 'Experiments') api<Reports>('/experiments').then(setReports).catch(e => setError(e.message)); }, [tab, loadQueue]);
  useEffect(() => { const close = (e: KeyboardEvent) => { if (e.key === 'Escape') { setPreview(null); if (!uploadBusy) setUploadOpen(false); } }; window.addEventListener('keydown', close); return () => window.removeEventListener('keydown', close); }, [uploadBusy]);

  async function search() {
    if (query.trim().length < 3 || searching) return;
    setSearching(true); setError(''); const start = performance.now();
    try {
      if (conversation) { const result = await api<{ results: Result[]; message: string }>('/chat', post({ message: query, history, method })); setResults(result.results); setHistory(h => [...h, query].slice(-10)); setTurns(t => [...t, { query, message: result.message }]); }
      else { const endpoint = method === 'semantic' ? '/search' : `/search/${method}`; const result = await api<{ results: Result[] }>(`${endpoint}?q=${encodeURIComponent(query)}&limit=5`); setResults(result.results); }
      setSearched(true); setElapsed(performance.now() - start); await refresh();
    } catch(e) { setError((e as Error).message); } finally { setSearching(false); }
  }
  async function saveLabel(item: Review) {
    setReviewBusy(true); setError('');
    try { await api('/review/labels', post(item)); setNote(''); setNotice(`Saved as ${pretty(item.label)}. Review priorities updated.`); await Promise.all([loadQueue(), refresh()]); }
    catch(e) { setError((e as Error).message); } finally { setReviewBusy(false); }
  }
  async function upload(e: FormEvent) {
    e.preventDefault(); setUploadBusy(true); setError(''); setNotice('');
    try {
      if (uploadMode === 'text') { await api('/documents/upload-text', post({ title: textTitle, content: textContent, source_url: sourceUrl || null, license: license || null })); setTextContent(''); setTextTitle(''); setNotice('Text document indexed.'); }
      else {
        const files = Array.from(fileRef.current?.files || []); if (!files.length) throw new Error('Select at least one file.');
        const outcomes: string[] = []; let failures = 0;
        for (const file of files) {
          try {
            const data = new FormData(); data.append('file', file);
            const result = await api<{ document_id: number; passages_created?: number; figures_created?: number; ocr?: {status:string;message:string|null;text:string} }>(file.name.toLowerCase().endsWith('.pdf') ? '/documents/upload' : '/documents/upload-image', { method: 'POST', body: data });
            outcomes.push(`${file.name}: ${result.passages_created || 0} passages, ${result.figures_created || 0} figures`);
            if (result.ocr) outcomes.push(result.ocr.status === 'extracted' ? `OCR: ${result.ocr.text.length} characters extracted and indexed. Open Image text in Library to read it.` : `OCR: ${result.ocr.message || result.ocr.status}`);
            if (sourceUrl || license) { try { await api(`/documents/${result.document_id}/provenance`, { ...post({ source_url: sourceUrl || null, license: license || null }), method: 'PATCH' }); } catch(e) { failures++; outcomes.push(`Source indexed, but provenance failed: ${(e as Error).message}`); } }
          } catch(e) { failures++; outcomes.push(`${file.name}: ${(e as Error).message}`); }
          setNotice(outcomes.join('\n'));
        }
        if (failures) { await refresh(); throw new Error(`${failures} operation(s) failed. See upload results.`); }
      }
      await refresh(); setUploadOpen(false); setTab('Library');
    } catch(e) { setError((e as Error).message); } finally { setUploadBusy(false); }
  }
  async function attach(doc: Doc, file?: File) {
    if (!file) return; setUploadBusy(true); setError(''); setNotice(`Attaching ${doc.filename} and indexing figures…`);
    try { const body = new FormData(); body.append('file', file); const r = await api<{ figures_created: number }>(`/documents/${doc.id}/source`, { method: 'POST', body }); setNotice(`Source attached; ${r.figures_created} figures indexed.`); await refresh(); }
    catch(e) { setError((e as Error).message); } finally { setUploadBusy(false); }
  }
  const current = queue?.results[0];
  return <div className="app-shell">
    <aside className="sidebar"><a aria-label="Atlas home" className="brand" href="#" onClick={e => { e.preventDefault(); setTab('Search'); }}><span className="brand-mark"><Layers size={23}/></span><span>atlas<small>DOCUMENT INTELLIGENCE</small></span></a><div className="workspace-label">WORKSPACE <span>LOCAL</span></div>
      <nav aria-label="Main navigation">{([{ name: 'Search', icon: Search, sub: 'Explore your knowledge' }, { name: 'Library', icon: BookOpen, sub: 'Documents & figures' }, { name: 'Review', icon: ShieldCheck, sub: 'Human in the loop' }, { name: 'Experiments', icon: FlaskConical, sub: 'Measure what matters' }] as const).map(({ name, icon: Icon, sub }) => <button aria-label={name} title={name} key={name} className={`nav-item ${tab === name ? 'active' : ''}`} onClick={() => { setTab(name); setError(''); }}><Icon size={20}/><span>{name}<small>{sub}</small></span>{tab === name && <i/>}</button>)}</nav>
      <div className="sidebar-bottom"><div className="model-status"><i/> FROZEN EMBEDDINGS</div><p>Sentence-BERT + CLIP<br/>Your knowledge, connected.</p><div className="profile"><span>MR</span><div>Research workspace<small>Multimodal Retrieval Engine</small></div></div></div>
    </aside>
    <main><header className="topbar"><div>Workspace <ChevronRight size={14}/><strong>{tab}</strong></div><span>RESEARCH PROTOTYPE</span><button className="button primary small" onClick={() => setUploadOpen(true)}><Plus size={16}/> Add documents</button></header><div className="page">
      {error && <div role="alert" className="alert error">{error}<button aria-label="Dismiss error" onClick={() => setError('')}><X size={16}/></button></div>}
      {notice && <div role="status" className="alert success">{notice}<button aria-label="Dismiss notification" onClick={() => setNotice('')}><X size={16}/></button></div>}
      {tab === 'Search' && <>
        <div className="eyebrow"><i/> CONNECT THE DOTS</div><div className="page-heading"><div><h1>Your next insight starts here.</h1><p>Search across passages, papers, and figures. Follow the evidence.</p></div><span className="outline-badge"><Layers size={14}/> MULTIMODAL SEARCH</span></div>
        <div className="stats-row">{[['Documents indexed', stats?.counts.documents, FileText], ['Text passages', stats?.counts.passages, Layers], ['Figures extracted', stats?.counts.figures, ImageIcon], ['Human labels', stats?.counts.labels, ShieldCheck]].map(([title, value, Icon]) => { const I = Icon as typeof FileText; return <div className="stat" key={String(title)}><I size={19}/><div><strong>{value == null ? '—' : Number(value).toLocaleString()}</strong><span>{String(title)}</span></div></div>; })}</div>
        <section className="search-panel"><div className="search-panel-title"><span><Search size={17}/> Explore your corpus</span><button disabled={searching} className={`toggle ${conversation ? 'selected' : ''}`} onClick={() => { setConversation(!conversation); setHistory([]); setTurns([]); }}><MessageSquare size={14}/> Conversation {conversation ? 'on' : 'off'}</button></div>
          <form className="search-box" onSubmit={e => { e.preventDefault(); search(); }}><Search size={22}/><input aria-label="Search documents" placeholder={conversation && history.length ? 'Refine your search…' : 'Ask a question or describe what you’re looking for…'} value={query} onChange={e => setQuery(e.target.value)} maxLength={500}/><button aria-label="Run search" className="button primary" disabled={searching || query.trim().length < 3}>{searching ? <LoaderCircle className="spin" size={18}/> : <>Search <ArrowUpRight size={16}/></>}</button></form>
          <div className="search-options"><div className="method-tabs">{Object.entries(methodNames).map(([key, name]) => <button disabled={searching} className={method === key ? 'selected' : ''} key={key} onClick={() => setMethod(key as Method)}>{name}</button>)}</div><span>Top 5 results</span></div>{conversation && <p className="helper">Refinements include your previous two messages. Results cite retrieved sources.</p>}
        </section>
        {turns.length > 0 && <div className="conversation-log">{turns.map((t, i) => <div key={i}><strong>{t.query}</strong><p>{t.message}</p></div>)}<button className="text-button" onClick={() => { setHistory([]); setTurns([]); }}>Start a new conversation</button></div>}
        {searched ? <><div className="section-heading"><h2>{results.length} sources</h2><span>{elapsed != null ? `${elapsed.toFixed(0)} ms · ` : ''}{methodNames[method]}</span></div><div className={method === 'images' ? 'figure-grid' : 'result-list'}>{results.map((item, i) => <article className="result-card" key={`${item.passage_id || item.figure_id || item.item_id}-${i}`}><div className="result-top"><span className="result-index">{String(i + 1).padStart(2, '0')}</span><span className="document-name">{item.filename}</span><span className="pill">{item.preview_url ? 'FIGURE' : 'PASSAGE'}</span></div>{item.preview_url && <img className="result-image" src={item.preview_url} alt={item.content || 'Extracted figure'}/>}<p className="result-content">{item.content || 'Figure extracted from the source document.'}</p><div className="result-footer"><span>Page {item.page_number || '—'}</span><span>{item.preview_url && <button className="text-button" onClick={() => setOcrTarget({documentId:item.document_id,filename:item.filename,figureId:item.figure_id||item.item_id})}>Extract image text · </button>}<button className="text-button" onClick={() => setPreview(item)}>View source <ArrowUpRight size={15}/></button></span></div></article>)}</div>{!results.length && <div className="empty-state"><Search/><h3>No matches found</h3><p>Try a broader question or add documents to your library.</p></div>}</> : <>
          <div className="section-heading"><h2>A few places to start</h2><span>ASK IN YOUR OWN WORDS</span></div><div className="suggestion-grid">{[{ icon: Layers, tag: 'CONCEPTS & METHODS', text: 'How do DFT, DCT and DST represent frequency components?' }, { icon: ShieldCheck, tag: 'CONNECT IDEAS', text: 'How does active learning prioritize uncertain examples?' }, { icon: ImageIcon, tag: 'VISUAL DISCOVERY', text: 'Find diagrams of neural network architectures' }].map(({ icon: Icon, tag, text }) => <button className="suggestion" key={tag} onClick={() => setQuery(text)}><Icon size={22}/><span>{tag}</span><strong>{text}</strong><ArrowUpRight size={18}/></button>)}</div><div className="how-it-works"><span className="icon-tile"><Layers size={24}/></span><div><h3>One workspace. More ways to discover.</h3><p>Semantic text search understands meaning. CLIP connects descriptions to figures.<br/>Human review helps a lightweight classifier learn where to look next.</p></div><span className="outline-badge">BUILT FOR RESEARCH</span></div>
        </>}
      </>}
      {tab === 'Library' && <><div className="eyebrow"><i/> YOUR KNOWLEDGE BASE</div><div className="page-heading"><div><h1>The source of your insights.</h1><p>Original documents, extracted passages, and visual evidence.</p></div><button className="button primary" onClick={() => setUploadOpen(true)}><Upload size={17}/> Add documents</button></div><div className="section-heading"><h2>{docs.length} documents</h2><input aria-label="Filter documents" className="filter-input" placeholder="Filter by filename…" value={docFilter} onChange={e => setDocFilter(e.target.value)}/></div>
        <div className="library-table"><div className="library-table-head"><span>DOCUMENT</span><span>SOURCE & LICENSE</span><span>STATUS</span><span>ACTIONS</span></div>{docs.filter(d => `${d.title} ${d.filename}`.toLowerCase().includes(docFilter.toLowerCase())).map(doc => <div className="library-row" key={doc.id}><div className="doc-cell"><span className="doc-icon"><FileText size={20}/></span><span><strong>{doc.title}</strong><small>{doc.filename} · #{doc.id}</small></span></div><div><small>{doc.license || 'License not recorded'}</small>{doc.source_url && <a href={doc.source_url} target="_blank" rel="noreferrer">Original source <ArrowUpRight size={12}/></a>}</div><span className={`status-pill ${doc.status === 'processed' ? 'ready' : ''}`}>{doc.status}</span><div className="row-actions">{doc.has_source && <button className="text-button" onClick={() => setOcrTarget({documentId:doc.id,filename:doc.filename})}>Image text</button>}{doc.has_source && <button className="text-button" onClick={() => setMetadataDoc(doc)}>Edit source metadata</button>}{doc.has_source ? <button className="text-button" onClick={() => setPreview({ document_id: doc.id, filename: doc.filename, content: '', page_number: 1 })}>Preview <ArrowUpRight size={14}/></button> : <label role="button" tabIndex={uploadBusy ? -1 : 0} aria-label={`Attach original PDF for ${doc.filename}`} onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.currentTarget.querySelector("input")?.click(); } }} className={`text-button attach ${uploadBusy ? 'disabled' : ''}`}>Attach PDF<input type="file" accept=".pdf" disabled={uploadBusy} onChange={e => attach(doc, e.target.files?.[0])}/></label>}<button className="text-button" onClick={async () => { try { const passages = await api<(Result & { id: number })[]>(`/documents/${doc.id}/passages`); setResults(passages.map(p => ({ ...p, passage_id: p.id, document_id: doc.id, filename: doc.filename }))); setSearched(true); setElapsed(null); setTab('Search'); } catch(e) { setError((e as Error).message); } }}>Passages</button></div></div>)}</div>
        {!docs.length && <div className="empty-state"><BookOpen/><h3>Build your research library</h3><p>Upload a PDF, image, or text document to get started.</p></div>}<p className="helper">Earlier records need their original PDF attached before source previews and figure search are available. Attaching preserves existing passage IDs.</p>
      </>}
      {tab === 'Review' && <><div className="eyebrow"><i/> HUMAN IN THE LOOP</div><div className="page-heading"><div><h1>A little judgment goes a long way.</h1><p>Review uncertain items. Improve the classifier with every correction.</p></div><span className="outline-badge"><ShieldCheck size={14}/> {stats?.counts.labels || 0} ITEMS LABELED</span></div><div className="review-toolbar"><label>Review<select disabled={reviewBusy} value={itemType} onChange={e => { setQueue(null); queueRequest.current++; setItemType(e.target.value as ItemType); }}><option value="passage">Text passages</option><option value="figure">Figures</option></select></label><label>Acquisition<select disabled={reviewBusy} value={strategy} onChange={e => { setQueue(null); queueRequest.current++; setStrategy(e.target.value); }}><option value="entropy">Highest entropy</option><option value="margin">Smallest margin</option><option value="random">Random baseline</option></select></label><button className="text-button" onClick={() => setSkipped({passage:[],figure:[]})}>Reset skipped items</button><span>{queue?.remaining ?? '—'} awaiting review</span></div>
        {queue && !queue.classifier_ready && <div className="cold-start"><FlaskConical size={19}/><div><strong>Start with a few human labels</strong><p>Random selection until at least 4 labels across 2 classes are available for this modality. Only the lightweight classifier is trained.</p></div></div>}
        {current ? <div className="review-layout"><section className="review-card"><div className="result-top"><span className="pill">{itemType.toUpperCase()}</span><span>{current.filename} · page {current.page_number}</span></div>{current.preview_url && <img className="review-image" src={current.preview_url} alt="Figure awaiting review"/>}<p className="review-content">{current.content || 'Review the extracted figure.'}</p><button className="text-button" onClick={() => setPreview(current)}>Open source <ArrowUpRight size={14}/></button>{current.uncertainty != null && <div className="prediction"><span>Acquisition score <strong>{current.uncertainty.toFixed(3)}</strong></span><span>Predicted <strong>{pretty(current.prediction || '')}</strong></span></div>}</section><section className="decision-card"><h2>How would you label this?</h2><p>Judge the item in the context of your curated technical corpus.</p><textarea aria-label="Review note" placeholder="Optional note for the review history…" value={note} onChange={e => setNote(e.target.value)} maxLength={2000}/><button className="text-button" disabled={reviewBusy || skipped[itemType].length >= 500} onClick={() => { setNote(''); setQueue(null); queueRequest.current++; setSkipped(s => ({...s,[itemType]:[...s[itemType],current.item_id!]})); }}>Skip uncertain item for now</button>{labels.map(l => <button className="label-button" key={l} disabled={reviewBusy} onClick={() => saveLabel({ item_type: itemType, item_id: current.item_id!, label: l, note })}><span>{pretty(l)}</span>{reviewBusy ? <LoaderCircle size={15} className="spin"/> : <ChevronRight size={15}/>}</button>)}</section></div> : <div className="empty-state">{queue ? <><Check/><h3>Queue complete</h3><p>Add more {itemType === 'figure' ? 'figures' : 'documents'} to continue reviewing.</p></> : <><LoaderCircle className="spin"/><p>Loading review queue…</p></>}</div>}
        {!!recent.length && <><div className="section-heading"><h2>Recent labels</h2><span>CORRECTIONS KEEP AN AUDIT HISTORY</span></div><div className="recent-labels">{recent.map(item => <div key={`${item.item_type}-${item.item_id}`}><span>{item.item_type} #{item.item_id}</span><select aria-label={`Correct label for ${item.item_type} ${item.item_id}`} disabled={reviewBusy} value={item.label} onChange={e => saveLabel({ ...item, label: e.target.value })}>{labels.map(l => <option value={l} key={l}>{pretty(l)}</option>)}</select><small>{item.note}</small></div>)}</div></>}
      </>}
      {tab === 'Experiments' && <><div className="eyebrow"><i/> EVIDENCE OVER ASSUMPTIONS</div><div className="page-heading"><div><h1>Measure what matters.</h1><p>Retrieval quality, label efficiency, and the speed–recall trade-off.</p></div><button className="button secondary" onClick={() => { api<Reports>('/experiments').then(setReports).catch(e => setError(e.message)); refresh().catch(e => setError(e.message)); }}>Refresh results</button></div><div className="metrics-grid">{[['Median search latency', stats?.search_latency.median_ms], ['95th percentile', stats?.search_latency.p95_ms]].map(([name, value]) => <div className="metric-card" key={String(name)}><BarChart3 size={20}/><span>{name}</span><strong>{value == null ? '—' : `${Number(value).toFixed(1)} ms`}</strong><small>{stats?.search_latency.sample_count || 0} successful requests · current server process</small></div>)}</div>
        <ResearchPanel/><div className="experiment-list">{[{ key: 'retrieval_summary', title: '01 / Retrieval effectiveness', description: 'Keyword, Sentence-BERT, hybrid, and CLIP retrieval. Precision@5, Recall@5, and MRR.', command: 'PYTHONPATH=backend .venv/bin/python experiments/evaluate_retrieval.py' }, { key: 'active_learning', title: '02 / Label efficiency', description: 'Entropy and margin versus random selection under equal budgets, using a fixed held-out set.', command: 'PYTHONPATH=backend .venv/bin/python experiments/evaluate_active_learning.py' }, { key: 'index_benchmark', title: '03 / Index performance', description: 'Exact versus HNSW at increasing corpus sizes. Median, p95 latency, and Recall@K.', command: 'PYTHONPATH=backend .venv/bin/python experiments/benchmark_index.py' }].map(e => <section className="experiment-card" key={e.key}><div><h2>{e.title}</h2><span className="status-pill">{reports?.reports[e.key] ? 'Report available' : 'Awaiting experiment'}</span></div><p>{e.description}</p><Report data={reports?.reports[e.key]}/><details><summary>Run this experiment</summary><p className="helper">Run from the repository root. See README for labels, judgments, and options.</p><code>{e.command}</code></details></section>)}</div><p className="helper">Experiments use real corpus data. Verify provisional judgments before reporting conclusions. Missing experiments have no generated scores.</p>
      </>}
      <footer>ATLAS / MULTIMODAL RETRIEVAL <span>Search with context. Review with intent.</span></footer>
    </div></main>
    {ocrTarget && <ImageTextDialog {...ocrTarget} close={() => setOcrTarget(null)} saved={refresh}/>}
    {metadataDoc && <ProvenanceEditor document={metadataDoc} close={() => setMetadataDoc(null)} saved={refresh}/>}
    {uploadOpen && <div className="modal-backdrop" onClick={() => { if (!uploadBusy) setUploadOpen(false); }}><section className="modal" role="dialog" aria-modal="true" aria-labelledby="upload-title" onClick={e => e.stopPropagation()}><button className="close-button" disabled={uploadBusy} aria-label="Close upload" onClick={() => setUploadOpen(false)}><X/></button><span className="icon-tile"><Upload/></span><h2 id="upload-title">Add to your knowledge base</h2><p>Index technical documents and figures with frozen embedding models.</p><div className="method-tabs"><button onClick={() => setUploadMode('files')} className={uploadMode === 'files' ? 'selected' : ''}>PDFs & images</button><button onClick={() => setUploadMode('text')} className={uploadMode === 'text' ? 'selected' : ''}>Text passage</button></div>
      {error && <div role="alert" className="alert error">{error}</div>}{notice && <div role="status" className="alert success">{notice}</div>}
      <form onSubmit={upload}>{uploadMode === 'files' ? <label className="dropzone"><Upload size={26}/><strong>Select source files</strong><span>PDF, PNG, JPEG or WebP · multiple files supported</span><input ref={fileRef} type="file" accept=".pdf,.png,.jpg,.jpeg,.webp" multiple required disabled={uploadBusy}/></label> : <><label>Document title<input value={textTitle} onChange={e => setTextTitle(e.target.value)} maxLength={200} required/></label><label>Text content<textarea value={textContent} onChange={e => setTextContent(e.target.value)} maxLength={1000000} required rows={7}/></label></>}<label>Original source URL <small>(optional)</small><input type="url" placeholder="https://…" value={sourceUrl} onChange={e => setSourceUrl(e.target.value)}/></label><label>Reuse license <small>(optional)</small><input value={license} onChange={e => setLicense(e.target.value)} placeholder="e.g. CC BY 4.0" maxLength={200}/></label><p className="helper">Add material you have permission to use. First model use may require a download. Image and scanned-PDF text extraction uses Tesseract. Uploaded image text is indexed automatically.</p><button className="button primary full" disabled={uploadBusy}>{uploadBusy ? <><LoaderCircle size={17} className="spin"/> Extracting & indexing…</> : <><Plus size={17}/> Add & index</>}</button></form></section></div>}
    {preview && <div className="modal-backdrop" onClick={() => setPreview(null)}><section className="modal preview-modal" role="dialog" aria-modal="true" aria-labelledby="preview-title" onClick={e => e.stopPropagation()}><button className="close-button" aria-label="Close preview" onClick={() => setPreview(null)}><X/></button><h2 id="preview-title">{preview.filename}</h2><p>Source evidence · page {preview.page_number || 1}</p>{docs.find(d => d.id === preview.document_id)?.has_source ? <><iframe title="Original source preview" src={`/api/v1/documents/${preview.document_id}/source#page=${preview.page_number || 1}`}/><a className="text-button" href={`/api/v1/documents/${preview.document_id}/source#page=${preview.page_number || 1}`} target="_blank" rel="noreferrer">Open original source <ArrowUpRight size={15}/></a></> : <div className="empty-state"><FileText/><h3>Original source needed</h3><p>Attach the original PDF from the Library to enable previews and figure extraction.</p><button className="button primary" onClick={() => { setPreview(null); setTab('Library'); }}>Go to Library</button></div>}</section></div>}
  </div>;
}

function Report({ data }: { data: unknown }) {
  if (!data) return <div className="report-empty"><FlaskConical size={18}/> Run the experiment to see measured results here.</div>;
  const r = data as Record<string, unknown>;
  if (r.macro_averages) return <><div className="report-note">{r.provisional_relevance_judgments ? 'PROVISIONAL JUDGMENTS · ' : ''}{String(r.query_count)} queries · {String(r.failure_count || 0)} failures</div><table className="metrics-table"><thead><tr><th>Method</th><th>P@5</th><th>R@5</th><th>MRR</th></tr></thead><tbody>{Object.entries(r.macro_averages as Record<string, Record<string, number>>).map(([m,v]) => <tr key={m}><td>{m}</td><td>{v.precision_at_5?.toFixed(3)}</td><td>{v.recall_at_5?.toFixed(3)}</td><td>{v.mrr?.toFixed(3)}</td></tr>)}</tbody></table></>;
  if (r.curves) return <><div className="report-note">HELD-OUT MACRO F1 · {String(r.heldout_count)} items · {String(r.seeds)} seeds</div><table className="metrics-table"><thead><tr><th>Selection</th><th>Labels</th><th>Macro F1</th><th>Std.</th></tr></thead><tbody>{(r.curves as { strategy: string; labels_reviewed: number; mean_f1: number; std_f1: number }[]).map((v,i) => <tr key={i}><td>{v.strategy}</td><td>{v.labels_reviewed}</td><td>{v.mean_f1.toFixed(3)}</td><td>{v.std_f1.toFixed(3)}</td></tr>)}</tbody></table></>;
  if (r.measurements) return <table className="metrics-table"><thead><tr><th>Corpus</th><th>Index</th><th>Median</th><th>p95</th><th>Recall@K</th></tr></thead><tbody>{(r.measurements as { corpus_size: number; mode: string; median_ms: number; p95_ms: number; recall_at_k: number }[]).map((v,i) => <tr key={i}><td>{v.corpus_size}</td><td>{v.mode}</td><td>{v.median_ms.toFixed(2)} ms</td><td>{v.p95_ms.toFixed(2)} ms</td><td>{v.recall_at_k.toFixed(3)}</td></tr>)}</tbody></table>;
  return <pre className="raw-report">{JSON.stringify(data, null, 2)}</pre>;
}
