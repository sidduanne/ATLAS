import { useRef, useState, type FormEvent } from 'react';
import { LoaderCircle, Plus, Upload } from 'lucide-react';
import Modal from '../components/Modal';
import { api, json, message, upload, type OcrRecord } from '../api';

type Uploaded = { document_id: number; passages_created?: number; figures_created?: number; ocr?: OcrRecord };

type Props = {
  onClose: () => void;
  /** Called after every attempt so the library reflects partial successes. */
  refresh: () => Promise<void>;
  /** Called when everything succeeded, with a summary for the user. */
  onDone: (summary: string) => void;
};

export default function UploadDialog({ onClose, refresh, onDone }: Props) {
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
