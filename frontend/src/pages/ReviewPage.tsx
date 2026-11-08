import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowUpRight, Check, ChevronRight, FlaskConical, LoaderCircle, ShieldCheck } from 'lucide-react';
import PageHeading from '../components/PageHeading';
import { api, json, LABELS, message, pretty, type ItemType, type Notify, type Result } from '../api';

type Queue = { results: Result[]; classifier_ready: boolean; strategy: string; labeled_count: number; remaining: number };
type Label = { item_type: ItemType; item_id: number; label: string; note: string };

type Props = {
  active: boolean;
  labeledCount: number;
  refresh: () => Promise<void>;
  notify: Notify;
  openPreview: (item: Result) => void;
};

export default function ReviewPage({ active, labeledCount, refresh, notify, openPreview }: Props) {
  const [itemType, setItemType] = useState<ItemType>('passage');
  const [strategy, setStrategy] = useState('entropy');
  const [skipped, setSkipped] = useState<Record<ItemType, number[]>>({ passage: [], figure: [] });
  const [queue, setQueue] = useState<Queue | null>(null);
  const [recent, setRecent] = useState<Label[]>([]);
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  // Ignore responses from queue requests that were superseded by a newer one.
  const latestRequest = useRef(0);

  const loadQueue = useCallback(async () => {
    const request = ++latestRequest.current;
    const exclude = skipped[itemType].map(id => `&exclude=${id}`).join('');
    const [nextQueue, labels] = await Promise.all([
      api<Queue>(`/review/queue?item_type=${itemType}&strategy=${strategy}${exclude}`),
      api<Label[]>('/review/labels?limit=15'),
    ]);
    if (request === latestRequest.current) { setQueue(nextQueue); setRecent(labels); }
  }, [itemType, strategy, skipped]);

  useEffect(() => {
    if (!active) return;
    setQueue(null);
    loadQueue().catch(e => notify.error(message(e)));
  }, [active, loadQueue]); // eslint-disable-line react-hooks/exhaustive-deps

  /** Clear the queue immediately when the selection changes, so stale items can't be labeled. */
  function changeQueue(update: () => void) {
    setQueue(null);
    latestRequest.current++;
    update();
  }

  async function save(label: Label) {
    setBusy(true);
    notify.error('');
    try {
      await api('/review/labels', json(label));
      setNote('');
      notify.notice(`Saved as ${pretty(label.label)}. Review priorities updated.`);
      await Promise.all([loadQueue(), refresh()]);
    } catch (e) {
      notify.error(message(e));
    } finally {
      setBusy(false);
    }
  }

  const current = queue?.results[0];
  return (
    <>
      <PageHeading eyebrow="HUMAN IN THE LOOP" title="A little judgment goes a long way."
        subtitle="Review uncertain items. Improve the classifier with every correction.">
        <span className="outline-badge"><ShieldCheck size={14} /> {labeledCount} ITEMS LABELED</span>
      </PageHeading>

      <div className="review-toolbar">
        <label>Review
          <select disabled={busy} value={itemType} onChange={e => changeQueue(() => setItemType(e.target.value as ItemType))}>
            <option value="passage">Text passages</option><option value="figure">Figures</option>
          </select>
        </label>
        <label>Acquisition
          <select disabled={busy} value={strategy} onChange={e => changeQueue(() => setStrategy(e.target.value))}>
            <option value="entropy">Highest entropy</option><option value="margin">Smallest margin</option><option value="random">Random baseline</option>
          </select>
        </label>
        <button className="text-button" onClick={() => setSkipped({ passage: [], figure: [] })}>Reset skipped items</button>
        <span>{queue?.remaining ?? '—'} awaiting review</span>
      </div>

      {queue && !queue.classifier_ready && (
        <div className="cold-start">
          <FlaskConical size={19} />
          <div>
            <strong>Start with a few human labels</strong>
            <p>Random selection until at least 4 labels across 2 classes are available for this modality. Only the lightweight classifier is trained.</p>
          </div>
        </div>
      )}

      {current ? (
        <div className="review-layout">
          <section className="review-card">
            <div className="result-top"><span className="pill">{itemType.toUpperCase()}</span><span>{current.filename} · page {current.page_number}</span></div>
            {current.preview_url && <img className="review-image" src={current.preview_url} alt="Figure awaiting review" />}
            <p className="review-content">{current.content || 'Review the extracted figure.'}</p>
            <button className="text-button" onClick={() => openPreview(current)}>Open source <ArrowUpRight size={14} /></button>
            {current.uncertainty != null && (
              <div className="prediction">
                <span>Acquisition score <strong>{current.uncertainty.toFixed(3)}</strong></span>
                <span>Predicted <strong>{pretty(current.prediction || '')}</strong></span>
              </div>
            )}
          </section>
          <section className="decision-card">
            <h2>How would you label this?</h2>
            <p>Judge the item in the context of your curated technical corpus.</p>
            <textarea aria-label="Review note" placeholder="Optional note for the review history…" value={note}
              onChange={e => setNote(e.target.value)} maxLength={2000} />
            <button className="text-button" disabled={busy || skipped[itemType].length >= 500}
              onClick={() => { setNote(''); changeQueue(() => setSkipped(s => ({ ...s, [itemType]: [...s[itemType], current.item_id!] }))); }}>
              Skip uncertain item for now
            </button>
            {LABELS.map(label => (
              <button className="label-button" key={label} disabled={busy}
                onClick={() => save({ item_type: itemType, item_id: current.item_id!, label, note })}>
                <span>{pretty(label)}</span>{busy ? <LoaderCircle size={15} className="spin" /> : <ChevronRight size={15} />}
              </button>
            ))}
          </section>
        </div>
      ) : (
        <div className="empty-state">
          {queue
            ? <><Check /><h3>Queue complete</h3><p>Add more {itemType === 'figure' ? 'figures' : 'documents'} to continue reviewing.</p></>
            : <><LoaderCircle className="spin" /><p>Loading review queue…</p></>}
        </div>
      )}

      {!!recent.length && <>
        <div className="section-heading"><h2>Recent labels</h2><span>CORRECTIONS KEEP AN AUDIT HISTORY</span></div>
        <div className="recent-labels">
          {recent.map(item => (
            <div key={`${item.item_type}-${item.item_id}`}>
              <span>{item.item_type} #{item.item_id}</span>
              <select aria-label={`Correct label for ${item.item_type} ${item.item_id}`} disabled={busy} value={item.label}
                onChange={e => save({ ...item, label: e.target.value })}>
                {LABELS.map(label => <option value={label} key={label}>{pretty(label)}</option>)}
              </select>
              <small>{item.note}</small>
            </div>
          ))}
        </div>
      </>}
    </>
  );
}
