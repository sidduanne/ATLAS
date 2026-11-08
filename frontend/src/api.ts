export type Tab = 'Search' | 'Library' | 'Review' | 'Experiments';
export type Method = 'hybrid' | 'semantic' | 'keyword' | 'images';
export type ItemType = 'passage' | 'figure';

export type Result = {
  passage_id?: number;
  figure_id?: number;
  item_id?: number;
  document_id: number;
  filename: string;
  content: string;
  page_number: number | null;
  preview_url?: string | null;
  uncertainty?: number | null;
  prediction?: string;
};
export type Doc = {
  id: number;
  filename: string;
  title: string;
  status: string;
  has_source: boolean;
  source_url: string | null;
  license: string | null;
};
export type Analytics = {
  counts: Record<string, number>;
  search_latency: { median_ms: number | null; p95_ms: number | null; sample_count: number };
};
export type OcrRecord = { figure_id: number; text: string; status: string; message: string | null };
/** A document (and optionally one of its figures) whose image text is being viewed. */
export type OcrTarget = { documentId: number; filename: string; figureId?: number };
export type Notify = { error: (message: string) => void; notice: (message: string) => void };

export const LABELS = ['relevant', 'duplicate', 'malformed', 'out_of_domain', 'irrelevant'];
export const METHOD_NAMES: Record<Method, string> = {
  hybrid: 'Hybrid search',
  semantic: 'Semantic text',
  keyword: 'Keyword baseline',
  images: 'Figures · CLIP',
};

export const pretty = (value: string) => value.replaceAll('_', ' ');
export const message = (error: unknown) => (error as Error).message;
export const sourceUrl = (documentId: number, page?: number | null) =>
  `/api/v1/documents/${documentId}/source#page=${page || 1}`;

/** Fetch `/api/v1{path}` and return its JSON, throwing the API's `detail` message on failure. */
export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api/v1${path}`, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status})`);
  }
  return data as T;
}

export const json = (body: unknown, method = 'POST'): RequestInit => ({
  method,
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export function upload(file: File): RequestInit {
  const body = new FormData();
  body.append('file', file);
  return { method: 'POST', body };
}
