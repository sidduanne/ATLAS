import { useEffect, useState } from 'react';
import Modal from '../components/Modal';
import { api, message, type OcrRecord, type OcrTarget } from '../api';

type Figure = { item_id: number; page_number: number; preview_url: string };

function buttonLabel(busy: boolean, record?: OcrRecord) {
  if (busy) return 'Extracting & indexing…';
  if (record?.status === 'extracted') return 'Text extracted and searchable';
  if (record?.status === 'no_text') return 'No readable text detected';
  return record ? 'Retry text extraction' : 'Extract & index text';
}

export default function ImageTextDialog({ documentId, filename, figureId, onClose, saved }:
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
