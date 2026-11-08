import { useEffect, useId, type ReactNode } from 'react';
import { X } from 'lucide-react';

type Props = {
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
export default function Modal({ title, subtitle, icon, className = '', busy = false, onClose, children }: Props) {
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
