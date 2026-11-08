import { useState } from 'react';
import { ArrowUpRight, BookOpen, FileText, Upload } from 'lucide-react';
import PageHeading from '../components/PageHeading';
import { api, message, upload, type Doc, type Notify, type OcrTarget, type Result } from '../api';

type Props = {
  docs: Doc[];
  refresh: () => Promise<void>;
  notify: Notify;
  openUpload: () => void;
  openPreview: (item: Result) => void;
  openImageText: (target: OcrTarget) => void;
  openMetadata: (doc: Doc) => void;
  showPassages: (results: Result[]) => void;
};

export default function LibraryPage({ docs, refresh, notify, openUpload, openPreview, openImageText, openMetadata, showPassages }: Props) {
  const [filter, setFilter] = useState('');
  const [attaching, setAttaching] = useState(false);
  const visible = docs.filter(d => `${d.title} ${d.filename}`.toLowerCase().includes(filter.toLowerCase()));

  async function attach(doc: Doc, file?: File) {
    if (!file) return;
    setAttaching(true);
    notify.error('');
    notify.notice(`Attaching ${doc.filename} and indexing figures…`);
    try {
      const result = await api<{ figures_created: number }>(`/documents/${doc.id}/source`, upload(file));
      notify.notice(`Source attached; ${result.figures_created} figures indexed.`);
      await refresh();
    } catch (e) {
      notify.error(message(e));
    } finally {
      setAttaching(false);
    }
  }

  async function passages(doc: Doc) {
    try {
      const rows = await api<(Result & { id: number })[]>(`/documents/${doc.id}/passages`);
      showPassages(rows.map(p => ({ ...p, passage_id: p.id, document_id: doc.id, filename: doc.filename })));
    } catch (e) {
      notify.error(message(e));
    }
  }

  return (
    <>
      <PageHeading eyebrow="YOUR KNOWLEDGE BASE" title="The source of your insights."
        subtitle="Original documents, extracted passages, and visual evidence.">
        <button className="button primary" onClick={openUpload}><Upload size={17} /> Add documents</button>
      </PageHeading>
      <div className="section-heading">
        <h2>{docs.length} documents</h2>
        <input aria-label="Filter documents" className="filter-input" placeholder="Filter by filename…" value={filter}
          onChange={e => setFilter(e.target.value)} />
      </div>

      <div className="library-table">
        <div className="library-table-head"><span>DOCUMENT</span><span>SOURCE & LICENSE</span><span>STATUS</span><span>ACTIONS</span></div>
        {visible.map(doc => (
          <div className="library-row" key={doc.id}>
            <div className="doc-cell">
              <span className="doc-icon"><FileText size={20} /></span>
              <span><strong>{doc.title}</strong><small>{doc.filename} · #{doc.id}</small></span>
            </div>
            <div>
              <small>{doc.license || 'License not recorded'}</small>
              {doc.source_url && <a href={doc.source_url} target="_blank" rel="noreferrer">Original source <ArrowUpRight size={12} /></a>}
            </div>
            <span className={`status-pill ${doc.status === 'processed' ? 'ready' : ''}`}>{doc.status}</span>
            <div className="row-actions">
              {doc.has_source ? <>
                <button className="text-button" onClick={() => openImageText({ documentId: doc.id, filename: doc.filename })}>Image text</button>
                <button className="text-button" onClick={() => openMetadata(doc)}>Edit source metadata</button>
                <button className="text-button" onClick={() => openPreview({ document_id: doc.id, filename: doc.filename, content: '', page_number: 1 })}>
                  Preview <ArrowUpRight size={14} />
                </button>
              </> : (
                <label role="button" tabIndex={attaching ? -1 : 0} aria-label={`Attach original PDF for ${doc.filename}`}
                  className={`text-button attach ${attaching ? 'disabled' : ''}`}
                  onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.currentTarget.querySelector('input')?.click(); } }}>
                  Attach PDF<input type="file" accept=".pdf" disabled={attaching} onChange={e => attach(doc, e.target.files?.[0])} />
                </label>
              )}
              <button className="text-button" onClick={() => passages(doc)}>Passages</button>
            </div>
          </div>
        ))}
      </div>
      {!docs.length && (
        <div className="empty-state"><BookOpen /><h3>Build your research library</h3><p>Upload a PDF, image, or text document to get started.</p></div>
      )}
      <p className="helper">Earlier records need their original PDF attached before source previews and figure search are available. Attaching preserves existing passage IDs.</p>
    </>
  );
}
