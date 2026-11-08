import { useState, type FormEvent } from 'react';
import Modal from '../components/Modal';
import { api, json, message, type Doc } from '../api';

export default function ProvenanceEditor({ document, onClose, saved }: { document: Doc; onClose: () => void; saved: () => Promise<void> }) {
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
