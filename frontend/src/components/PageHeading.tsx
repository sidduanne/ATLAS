import type { ReactNode } from 'react';

/** Eyebrow, title, subtitle and an optional right-hand action shared by every page. */
export default function PageHeading({ eyebrow, title, subtitle, children }:
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
