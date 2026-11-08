import { useState } from 'react';
import { ArrowUpRight, FileText, Image as ImageIcon, Layers, LoaderCircle, MessageSquare, Search, ShieldCheck } from 'lucide-react';
import { PageHeading } from './ui';
import { api, json, message, METHOD_NAMES, type Analytics, type Method, type Notify, type OcrTarget, type Result } from './api';

/** Results currently shown: a search, or a document's passages opened from Library. */
export type ResultsView = { results: Result[]; elapsed: number | null };

type Props = {
  stats: Analytics | null;
  view: ResultsView | null;
  setView: (view: ResultsView) => void;
  refresh: () => Promise<void>;
  notify: Notify;
  openPreview: (item: Result) => void;
  openImageText: (target: OcrTarget) => void;
};

const SUGGESTIONS = [
  { icon: Layers, tag: 'CONCEPTS & METHODS', text: 'How do DFT, DCT and DST represent frequency components?' },
  { icon: ShieldCheck, tag: 'CONNECT IDEAS', text: 'How does active learning prioritize uncertain examples?' },
  { icon: ImageIcon, tag: 'VISUAL DISCOVERY', text: 'Find diagrams of neural network architectures' },
];

export default function SearchPage({ stats, view, setView, refresh, notify, openPreview, openImageText }: Props) {
  const [query, setQuery] = useState('');
  const [method, setMethod] = useState<Method>('hybrid');
  const [searching, setSearching] = useState(false);
  const [conversation, setConversation] = useState(false);
  const [history, setHistory] = useState<string[]>([]);
  const [turns, setTurns] = useState<{ query: string; message: string }[]>([]);

  function resetConversation() { setHistory([]); setTurns([]); }

  async function search() {
    if (query.trim().length < 3 || searching) return;
    setSearching(true);
    notify.error('');
    const started = performance.now();
    try {
      let results: Result[];
      if (conversation) {
        const reply = await api<{ results: Result[]; message: string }>('/chat', json({ message: query, history, method }));
        results = reply.results;
        setHistory(h => [...h, query].slice(-10));
        setTurns(t => [...t, { query, message: reply.message }]);
      } else {
        const endpoint = method === 'semantic' ? '/search' : `/search/${method}`;
        results = (await api<{ results: Result[] }>(`${endpoint}?q=${encodeURIComponent(query)}&limit=5`)).results;
      }
      setView({ results, elapsed: performance.now() - started });
      await refresh();
    } catch (e) {
      notify.error(message(e));
    } finally {
      setSearching(false);
    }
  }

  const stat = [['Documents indexed', stats?.counts.documents, FileText], ['Text passages', stats?.counts.passages, Layers],
    ['Figures extracted', stats?.counts.figures, ImageIcon], ['Human labels', stats?.counts.labels, ShieldCheck]] as const;

  return (
    <>
      <PageHeading eyebrow="CONNECT THE DOTS" title="Your next insight starts here."
        subtitle="Search across passages, papers, and figures. Follow the evidence.">
        <span className="outline-badge"><Layers size={14} /> MULTIMODAL SEARCH</span>
      </PageHeading>

      <div className="stats-row">
        {stat.map(([title, value, Icon]) => (
          <div className="stat" key={title}>
            <Icon size={19} />
            <div><strong>{value == null ? '—' : Number(value).toLocaleString()}</strong><span>{title}</span></div>
          </div>
        ))}
      </div>

      <section className="search-panel">
        <div className="search-panel-title">
          <span><Search size={17} /> Explore your corpus</span>
          <button disabled={searching} className={`toggle ${conversation ? 'selected' : ''}`}
            onClick={() => { setConversation(!conversation); resetConversation(); }}>
            <MessageSquare size={14} /> Conversation {conversation ? 'on' : 'off'}
          </button>
        </div>
        <form className="search-box" onSubmit={event => { event.preventDefault(); search(); }}>
          <Search size={22} />
          <input aria-label="Search documents" value={query} onChange={e => setQuery(e.target.value)} maxLength={500}
            placeholder={conversation && history.length ? 'Refine your search…' : 'Ask a question or describe what you’re looking for…'} />
          <button aria-label="Run search" className="button primary" disabled={searching || query.trim().length < 3}>
            {searching ? <LoaderCircle className="spin" size={18} /> : <>Search <ArrowUpRight size={16} /></>}
          </button>
        </form>
        <div className="search-options">
          <div className="method-tabs">
            {(Object.entries(METHOD_NAMES) as [Method, string][]).map(([key, name]) => (
              <button key={key} disabled={searching} className={method === key ? 'selected' : ''} onClick={() => setMethod(key)}>{name}</button>
            ))}
          </div>
          <span>Top 5 results</span>
        </div>
        {conversation && <p className="helper">Refinements include your previous two messages. Results cite retrieved sources.</p>}
      </section>

      {turns.length > 0 && (
        <div className="conversation-log">
          {turns.map((turn, i) => <div key={i}><strong>{turn.query}</strong><p>{turn.message}</p></div>)}
          <button className="text-button" onClick={resetConversation}>Start a new conversation</button>
        </div>
      )}

      {view ? <Results view={view} method={method} openPreview={openPreview} openImageText={openImageText} /> : <Suggestions onPick={setQuery} />}
    </>
  );
}

function Results({ view, method, openPreview, openImageText }:
  { view: ResultsView; method: Method; openPreview: (item: Result) => void; openImageText: (target: OcrTarget) => void }) {
  const { results, elapsed } = view;
  return (
    <>
      <div className="section-heading">
        <h2>{results.length} sources</h2>
        <span>{elapsed != null ? `${elapsed.toFixed(0)} ms · ` : ''}{METHOD_NAMES[method]}</span>
      </div>
      <div className={method === 'images' ? 'figure-grid' : 'result-list'}>
        {results.map((item, i) => (
          <article className="result-card" key={`${item.passage_id || item.figure_id || item.item_id}-${i}`}>
            <div className="result-top">
              <span className="result-index">{String(i + 1).padStart(2, '0')}</span>
              <span className="document-name">{item.filename}</span>
              <span className="pill">{item.preview_url ? 'FIGURE' : 'PASSAGE'}</span>
            </div>
            {item.preview_url && <img className="result-image" src={item.preview_url} alt={item.content || 'Extracted figure'} />}
            <p className="result-content">{item.content || 'Figure extracted from the source document.'}</p>
            <div className="result-footer">
              <span>Page {item.page_number || '—'}</span>
              <span>
                {item.preview_url && (
                  <button className="text-button" onClick={() => openImageText({ documentId: item.document_id, filename: item.filename, figureId: item.figure_id || item.item_id })}>
                    Extract image text ·{' '}
                  </button>
                )}
                <button className="text-button" onClick={() => openPreview(item)}>View source <ArrowUpRight size={15} /></button>
              </span>
            </div>
          </article>
        ))}
      </div>
      {!results.length && (
        <div className="empty-state"><Search /><h3>No matches found</h3><p>Try a broader question or add documents to your library.</p></div>
      )}
    </>
  );
}

function Suggestions({ onPick }: { onPick: (text: string) => void }) {
  return (
    <>
      <div className="section-heading"><h2>A few places to start</h2><span>ASK IN YOUR OWN WORDS</span></div>
      <div className="suggestion-grid">
        {SUGGESTIONS.map(({ icon: Icon, tag, text }) => (
          <button className="suggestion" key={tag} onClick={() => onPick(text)}>
            <Icon size={22} /><span>{tag}</span><strong>{text}</strong><ArrowUpRight size={18} />
          </button>
        ))}
      </div>
      <div className="how-it-works">
        <span className="icon-tile"><Layers size={24} /></span>
        <div>
          <h3>One workspace. More ways to discover.</h3>
          <p>Semantic text search understands meaning. CLIP connects descriptions to figures.<br />Human review helps a lightweight classifier learn where to look next.</p>
        </div>
        <span className="outline-badge">BUILT FOR RESEARCH</span>
      </div>
    </>
  );
}
