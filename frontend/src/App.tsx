import { useCallback, useEffect, useMemo, useState } from 'react';
import { ChevronRight, Plus, X } from 'lucide-react';
import './style.css';
import { api, message, type Analytics, type Doc, type Notify, type OcrTarget, type Result, type Tab } from './api';
import Sidebar from './components/Sidebar';
import ImageTextDialog from './dialogs/ImageTextDialog';
import PreviewDialog from './dialogs/PreviewDialog';
import ProvenanceEditor from './dialogs/ProvenanceEditor';
import UploadDialog from './dialogs/UploadDialog';
import ExperimentsPage from './pages/ExperimentsPage';
import LibraryPage from './pages/LibraryPage';
import ReviewPage from './pages/ReviewPage';
import SearchPage, { type ResultsView } from './pages/SearchPage';

export default function App() {
  const [tab, setTab] = useState<Tab>('Search');
  const [stats, setStats] = useState<Analytics | null>(null);
  const [docs, setDocs] = useState<Doc[]>([]);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [view, setView] = useState<ResultsView | null>(null);
  // Open dialogs.
  const [uploading, setUploading] = useState(false);
  const [preview, setPreview] = useState<Result | null>(null);
  const [imageText, setImageText] = useState<OcrTarget | null>(null);
  const [metadataDoc, setMetadataDoc] = useState<Doc | null>(null);

  const notify: Notify = useMemo(() => ({ error: setError, notice: setNotice }), []);
  const refresh = useCallback(async () => {
    const [analytics, documents] = await Promise.all([api<Analytics>('/analytics'), api<Doc[]>('/documents')]);
    setStats(analytics);
    setDocs(documents);
  }, []);
  useEffect(() => { refresh().catch(e => setError(message(e))); }, [refresh]);

  function select(next: Tab) { setTab(next); setError(''); }

  return (
    <div className="app-shell">
      <Sidebar tab={tab} onSelect={select} />
      <main>
        <header className="topbar">
          <div>Workspace <ChevronRight size={14} /><strong>{tab}</strong></div>
          <span>RESEARCH PROTOTYPE</span>
          <button className="button primary small" onClick={() => setUploading(true)}><Plus size={16} /> Add documents</button>
        </header>
        <div className="page">
          {error && <div role="alert" className="alert error">{error}<button aria-label="Dismiss error" onClick={() => setError('')}><X size={16} /></button></div>}
          {notice && <div role="status" className="alert success">{notice}<button aria-label="Dismiss notification" onClick={() => setNotice('')}><X size={16} /></button></div>}

          {/* Search, Library and Review stay mounted so their state survives tab switches. */}
          <div hidden={tab !== 'Search'}>
            <SearchPage stats={stats} view={view} setView={setView} refresh={refresh} notify={notify}
              openPreview={setPreview} openImageText={setImageText} />
          </div>
          <div hidden={tab !== 'Library'}>
            <LibraryPage docs={docs} refresh={refresh} notify={notify} openUpload={() => setUploading(true)}
              openPreview={setPreview} openImageText={setImageText} openMetadata={setMetadataDoc}
              showPassages={results => { setView({ results, elapsed: null }); select('Search'); }} />
          </div>
          <div hidden={tab !== 'Review'}>
            <ReviewPage active={tab === 'Review'} labeledCount={stats?.counts.labels || 0} refresh={refresh} notify={notify}
              openPreview={setPreview} />
          </div>
          {tab === 'Experiments' && <ExperimentsPage stats={stats} refresh={refresh} notify={notify} />}

          <footer>ATLAS / MULTIMODAL RETRIEVAL <span>Search with context. Review with intent.</span></footer>
        </div>
      </main>

      {uploading && (
        <UploadDialog onClose={() => setUploading(false)} refresh={refresh}
          onDone={summary => { setNotice(summary); setUploading(false); select('Library'); }} />
      )}
      {preview && (
        <PreviewDialog item={preview} hasSource={!!docs.find(d => d.id === preview.document_id)?.has_source}
          onClose={() => setPreview(null)} openLibrary={() => { setPreview(null); select('Library'); }} />
      )}
      {imageText && <ImageTextDialog {...imageText} onClose={() => setImageText(null)} saved={refresh} />}
      {metadataDoc && <ProvenanceEditor document={metadataDoc} onClose={() => setMetadataDoc(null)} saved={refresh} />}
    </div>
  );
}
