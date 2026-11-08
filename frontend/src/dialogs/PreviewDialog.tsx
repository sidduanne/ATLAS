import { ArrowUpRight, FileText } from 'lucide-react';
import Modal from '../components/Modal';
import { sourceUrl, type Result } from '../api';

type Props = { item: Result; hasSource: boolean; onClose: () => void; openLibrary: () => void };

export default function PreviewDialog({ item, hasSource, onClose, openLibrary }: Props) {
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
