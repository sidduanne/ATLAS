import { useEffect, useRef, useState, type FormEvent } from 'react';
import { ArrowUpRight, FileText, LoaderCircle, Plus, Upload } from 'lucide-react';
import { Modal } from './ui';
import { api, json, message, sourceUrl, upload, type Doc, type OcrRecord, type OcrTarget, type Result } from './api';

type Uploaded = { document_id: number; passages_created?: number; figures_created?: number; ocr?: OcrRecord };

type UploadProps = {
  onClose: () => void;
  /** Called after every attempt so the library reflects partial successes. */
  refresh: () => Promise<void>;
  /** Called when everything succeeded, with a summary for the user. */
  onDone: (summary: string) => void;
};

export function UploadDialog({ onClose, refresh, onDone }: UploadProps) {
  const [mode, setMode] = useState<'files' | 'text'>('files');
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [sourceUrl, setSourceUrl] = useState('');
  const [license, setLicense] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [progress, setProgress] = useState('');
  const fileInput = useRef<HTMLInputElement>(null);
  const provenance = { source_url: sourceUrl || null, license: license || null };

  async function uploadFiles(): Promise<string> {
    const files = Array.from(fileInput.current?.files || []);
    if (!files.length) throw new Error('Select at least one file.');
    const lines: string[] = [];
    let failures = 0;
    for (const file of files) {
      try {
        const endpoint = file.name.toLowerCase().endsWith('.pdf') ? '/documents/upload' : '/documents/upload-image';
        const result = await api<Uploaded>(endpoint, upload(file));
        lines.push(`${file.name}: ${result.passages_created || 0} passages, ${result.figures_created || 0} figures`);
        if (result.ocr) {
          lines.push(result.ocr.status === 'extracted'
            ? `OCR: ${result.ocr.text.length} characters extracted and indexed. Open Image text in Library to read it.`
            : `OCR: ${result.ocr.message || result.ocr.status}`);
        }
        if (sourceUrl || license) {
          try {
            await api(`/documents/${result.document_id}/provenance`, json(provenance, 'PATCH'));
          } catch (e) {
            failures++;
            lines.push(`Source indexed, but provenance failed: ${message(e)}`);
          }
        }
      } catch (e) {
        failures++;
        lines.push(`${file.name}: ${message(e)}`);
      }
      setProgress(lines.join('\n'));
    }
    if (failures) throw new Error(`${failures} operation(s) failed. See upload results.`);
    return lines.join('\n');
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError('');
    setProgress('');
    try {
      let summary = 'Text document indexed.';
      if (mode === 'text') await api('/documents/upload-text', json({ title, content, ...provenance }));
      else summary = await uploadFiles();
      await refresh();
      onDone(summary);
    } catch (e) {
      setError(message(e));
      await refresh().catch(() => undefined);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal title="Add to your knowledge base" subtitle="Index technical documents and figures with frozen embedding models."
      icon={<Upload />} busy={busy} onClose={onClose}>
      <div className="method-tabs">
        <button onClick={() => setMode('files')} className={mode === 'files' ? 'selected' : ''}>PDFs & images</button>
        <button onClick={() => setMode('text')} className={mode === 'text' ? 'selected' : ''}>Text passage</button>
      </div>
      {error && <div role="alert" className="alert error">{error}</div>}
      {progress && <div role="status" className="alert success">{progress}</div>}
      <form onSubmit={submit}>
        {mode === 'files' ? (
          <label className="dropzone">
            <Upload size={26} /><strong>Select source files</strong><span>PDF, PNG, JPEG or WebP · multiple files supported</span>
            <input ref={fileInput} type="file" accept=".pdf,.png,.jpg,.jpeg,.webp" multiple required disabled={busy} />
          </label>
        ) : <>
          <label>Document title<input value={title} onChange={e => setTitle(e.target.value)} maxLength={200} required /></label>
          <label>Text content<textarea value={content} onChange={e => setContent(e.target.value)} maxLength={1000000} required rows={7} /></label>
        </>}
        <label>Original source URL <small>(optional)</small>
          <input type="url" placeholder="https://…" value={sourceUrl} onChange={e => setSourceUrl(e.target.value)} />
        </label>
        <label>Reuse license <small>(optional)</small>
          <input value={license} onChange={e => setLicense(e.target.value)} placeholder="e.g. CC BY 4.0" maxLength={200} />
        </label>
        <p className="helper">Add material you have permission to use. Image and scanned-PDF text extraction uses Tesseract. Uploaded image text is indexed automatically.</p>
        <button className="button primary full" disabled={busy}>
          {busy ? <><LoaderCircle size={17} className="spin" /> Extracting & indexing…</> : <><Plus size={17} /> Add & index</>}
        </button>
      </form>
    </Modal>
  );
}

type PreviewProps = { item: Result; hasSource: boolean; onClose: () => void; openLibrary: () => void };

export function PreviewDialog({ item, hasSource, onClose, openLibrary }: PreviewProps) {
  const url = sourceUrl(item.document_id, item.page_number);
  return (
    <Modal className="preview-modal" title={item.filename} subtitle={`Source evidence · page ${item.page_number || 1}`} onClose={onClose}>
      {hasSource ? <>
        <iframe title="Original source preview" src={url} />
        <a className="text-button" href={url} target="_blank" rel="noreferrer">Open original source <ArrowUpRight size={15} /></a>
      </> : (
        <div className="empty-state">
          <FileText />
          <h3>Original source needed</h3>
          <p>Attach the original PDF from the Library to enable previews and figure extraction.</p>
          <button className="button primary" onClick={openLibrary}>Go to Library</button>
        </div>
      )}
    </Modal>
  );
}

type Figure = { item_id: number; page_number: number; preview_url: string };

function buttonLabel(busy: boolean, record?: OcrRecord) {
  if (busy) return 'Extracting & indexing…';
  if (record?.status === 'extracted') return 'Text extracted and searchable';
  if (record?.status === 'no_text') return 'No readable text detected';
  return record ? 'Retry text extraction' : 'Extract & index text';
}

export function ImageTextDialog({ documentId, filename, figureId, onClose, saved }:
  OcrTarget & { onClose: () => void; saved: () => Promise<void> }) {
  const [figures, setFigures] = useState<Figure[] | null>(null);
  const [records, setRecords] = useState<OcrRecord[]>([]);
  const [selected, setSelected] = useState<number | null>(figureId || null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    Promise.all([api<Figure[]>(`/documents/${documentId}/figures`), api<{ results: OcrRecord[] }>(`/documents/${documentId}/ocr`)])
      .then(([images, ocr]) => {
        if (cancelled) return;
        setFigures(images);
        setRecords(ocr.results);
        setSelected(figureId || images[0]?.item_id || null);
      })
      .catch(e => { if (!cancelled) { setError(message(e)); setFigures([]); } });
    return () => { cancelled = true; };
  }, [documentId, figureId]);

  const figure = figures?.find(f => f.item_id === selected);
  const record = records.find(r => r.figure_id === selected);

  async function extract() {
    if (!selected) return;
    setBusy(true);
    setError('');
    try {
      const result = await api<OcrRecord>(`/figures/${selected}/ocr?retry=true`, { method: 'POST' });
      setRecords(r => [...r.filter(i => i.figure_id !== selected), result]);
      await saved();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal className="image-text-modal" title="Text from images" subtitle={filename} busy={busy} onClose={onClose}>
      {error && <div className="alert error" role="alert">{error}</div>}
      {figures === null ? <p role="status">Loading images…</p> : !figures.length ? <p>No extracted images are available for this document.</p> : <>
        <label>Image
          <select aria-label="Image to extract text from" value={selected || ''} disabled={busy} onChange={e => setSelected(Number(e.target.value))}>
            {figures.map(f => <option key={f.item_id} value={f.item_id}>Figure #{f.item_id} · page {f.page_number}</option>)}
          </select>
        </label>
        {figure && <img className="ocr-image" src={figure.preview_url} alt="Image for text extraction" />}
        <button className="button primary full" disabled={busy || record?.status === 'extracted' || record?.status === 'no_text'} onClick={extract}>
          {buttonLabel(busy, record)}
        </button>
        {record && <>
          <p role="status" className="helper">
            {record.message || (record.status === 'extracted'
              ? 'Search this text using semantic, keyword, or hybrid search. OCR can make mistakes; compare it with the image.'
              : record.status)}
          </p>
          <label>Extracted text<textarea aria-label="Extracted image text" readOnly rows={9} value={record.text} placeholder="No text extracted." /></label>
        </>}
        <p className="helper">English OCR runs locally. Uploaded images are processed automatically; PDF figure crops can be processed here on demand.</p>
      </>}
    </Modal>
  );
}

export function ProvenanceEditor({ document, onClose, saved }: { document: Doc; onClose: () => void; saved: () => Promise<void> }) {
  const [url, setUrl] = useState(document.source_url || '');
  const [license, setLicense] = useState(document.license || '');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    try {
      await api(`/documents/${document.id}/provenance`, json({ source_url: url || null, license: license || null }, 'PATCH'));
      await saved();
      onClose();
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal title="Source metadata" subtitle={document.filename} busy={busy} onClose={onClose}>
      {error && <div className="alert error">{error}</div>}
      <form onSubmit={submit}>
        <label>Original source URL<input type="url" value={url} onChange={e => setUrl(e.target.value)} /></label>
        <label>Reuse license<input maxLength={200} value={license} onChange={e => setLicense(e.target.value)} /></label>
        <p className="helper">Use the actual license published by the source. A hosting website alone does not establish permission to reuse.</p>
        <button className="button primary full" disabled={busy}>Save metadata</button>
      </form>
    </Modal>
  );
}
