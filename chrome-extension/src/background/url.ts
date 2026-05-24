// Canonicalize a URL down to origin + pathname (no hash, no trailing slash)
// so we can tell "user reloaded" from "user navigated to a new page".
// Query strings are preserved because most sites carry meaningful state
// there (article IDs, search terms) but YouTube watch URLs are normalized
// down to the video ID so ?t= timestamps don't spawn duplicate sessions.
export function canonicalize(raw: string): string {
  let u: URL
  try {
    u = new URL(raw)
  } catch {
    return raw
  }

  if (isYouTubeWatch(u)) {
    const v = u.searchParams.get("v")
    return v ? `https://www.youtube.com/watch?v=${v}` : `${u.origin}${u.pathname}`
  }

  let pathname = u.pathname
  if (pathname.length > 1 && pathname.endsWith("/")) {
    pathname = pathname.slice(0, -1)
  }
  return `${u.origin}${pathname}${u.search}`
}

function isYouTubeWatch(u: URL): boolean {
  return (
    /(^|\.)youtube\.com$/.test(u.hostname) && u.pathname === "/watch"
  )
}

const YT = /^https?:\/\/(?:www\.|m\.|music\.)?(?:youtube\.com\/(?:watch\?[^#]*\bv=[\w-]{11}|shorts\/[\w-]{11}|live\/[\w-]{11})|youtu\.be\/[\w-]{11})/i

export function isYouTube(url: string): boolean {
  return YT.test(url)
}

// Sites that should NOT spawn a Lloyd session.
//
// - google.com (any subdomain): search, mail, drive, calendar, etc.
//   None of these have summarizable "page content" in the usual sense,
//   and they'd produce a flood of sessions.
// - youtube.com (non-video pages): home, search, channel, /feed/*.
//   Video URLs (watch / shorts / live / youtu.be) are still allowed.
export function shouldSpawnSession(url: string): boolean {
  let u: URL
  try {
    u = new URL(url)
  } catch {
    return false
  }
  const host = u.hostname.toLowerCase()
  if (host === "google.com" || host.endsWith(".google.com")) {
    return false
  }
  if (host === "youtube.com" || host.endsWith(".youtube.com")) {
    return isYouTube(url)
  }
  return true
}
