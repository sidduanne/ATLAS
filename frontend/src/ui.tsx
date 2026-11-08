import { useEffect, useId, type ReactNode } from 'react';
import { BookOpen, FlaskConical, Layers, Search, ShieldCheck, X } from 'lucide-react';
import type { Tab } from './api';

type ModalProps = {
  title: string;
  subtitle?: ReactNode;
  icon?: ReactNode;
  className?: string;
  /** While busy, the dialog cannot be closed. */
  busy?: boolean;
  onClose: () => void;
  children: ReactNode;
};

/** Dialog with backdrop; closes on backdrop click, the close button, or Escape. */
export function Modal({ title, subtitle, icon, className = '', busy = false, onClose, children }: ModalProps) {
  const titleId = useId();
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape' && !busy) onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [busy, onClose]);

  return (
    <div className="modal-backdrop" onClick={() => { if (!busy) onClose(); }}>
      <section className={`modal ${className}`} role="dialog" aria-modal="true" aria-labelledby={titleId}
        onClick={event => event.stopPropagation()}>
        <button className="close-button" aria-label={`Close ${title}`} disabled={busy} onClick={onClose}><X /></button>
        {icon && <span className="icon-tile">{icon}</span>}
        <h2 id={titleId}>{title}</h2>
        {subtitle && <p>{subtitle}</p>}
        {children}
      </section>
    </div>
  );
}

/** Eyebrow, title, subtitle and an optional right-hand action shared by every page. */
export function PageHeading({ eyebrow, title, subtitle, children }:
  { eyebrow: string; title: string; subtitle: string; children?: ReactNode }) {
  return (
    <>
      <div className="eyebrow"><i /> {eyebrow}</div>
      <div className="page-heading">
        <div><h1>{title}</h1><p>{subtitle}</p></div>
        {children}
      </div>
    </>
  );
}

const NAV = [
  { name: 'Search', icon: Search, sub: 'Explore your knowledge' },
  { name: 'Library', icon: BookOpen, sub: 'Documents & figures' },
  { name: 'Review', icon: ShieldCheck, sub: 'Human in the loop' },
  { name: 'Experiments', icon: FlaskConical, sub: 'Measure what matters' },
] as const;

export function Sidebar({ tab, onSelect }: { tab: Tab; onSelect: (tab: Tab) => void }) {
  return (
    <aside className="sidebar">
      <a aria-label="Atlas home" className="brand" href="#" onClick={event => { event.preventDefault(); onSelect('Search'); }}>
        <span className="brand-mark"><Layers size={23} /></span>
        <span>atlas<small>DOCUMENT INTELLIGENCE</small></span>
      </a>
      <div className="workspace-label">WORKSPACE <span>LOCAL</span></div>
      <nav aria-label="Main navigation">
        {NAV.map(({ name, icon: Icon, sub }) => (
          <button key={name} aria-label={name} title={name} className={`nav-item ${tab === name ? 'active' : ''}`}
            onClick={() => onSelect(name)}>
            <Icon size={20} />
            <span>{name}<small>{sub}</small></span>
            {tab === name && <i />}
          </button>
        ))}
      </nav>
      <div className="sidebar-bottom">
        <div className="model-status"><i /> FROZEN EMBEDDINGS</div>
        <p>Sentence-BERT + CLIP<br />Your knowledge, connected.</p>
        <div className="profile">
          <span>MR</span>
          <div>Research workspace<small>Multimodal Retrieval Engine</small></div>
        </div>
      </div>
    </aside>
  );
}
