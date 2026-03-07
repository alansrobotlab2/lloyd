import DOMPurify from "dompurify";
import { marked } from "marked";

export function sanitizeHtml(md: string): string {
  return DOMPurify.sanitize(marked.parse(md) as string);
}
