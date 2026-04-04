import { marked } from "marked"

export function sanitizeHtml(md: string): string {
  return marked.parse(md) as string
}
