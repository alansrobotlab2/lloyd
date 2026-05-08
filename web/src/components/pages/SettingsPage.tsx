import { useCallback, useEffect, useRef, useState } from 'react'
import {
  Mic,
  Square,
  Trash2,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  Copy,
  Check,
  Download,
  Wifi,
  ShieldCheck,
  Plus,
} from 'lucide-react'
import { api } from '../../api'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

const RECORD_SECONDS = 5
const SAMPLE_RATE_HINT = 16000  // resemblyzer resamples internally; this is just a request

interface Profile {
  name: string
  embedding_dim: number
  path: string
}

/** Capture `seconds` of mic audio and return it as a 16-bit PCM WAV blob.
 *  Uses the Web Audio API directly (ScriptProcessor) to get raw samples;
 *  MediaRecorder produces webm/opus, which the backend's `wave.open` won't
 *  parse. Mono. Browser sample rate (typically 48 kHz) — resemblyzer's
 *  preprocess_wav handles the resample. */
async function recordWav(
  seconds: number,
  onProgress?: (elapsedMs: number) => void,
  signal?: AbortSignal,
): Promise<Blob> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Mic capture requires a secure context (https or http://localhost)")
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, sampleRate: SAMPLE_RATE_HINT, echoCancellation: true, noiseSuppression: true },
  })
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const AC = (window.AudioContext || (window as any).webkitAudioContext) as typeof AudioContext
  const ctx = new AC()
  const source = ctx.createMediaStreamSource(stream)
  // ScriptProcessorNode is deprecated but uniformly available; AudioWorklet
  // would need a separate worklet file. For a one-shot 5s record this is
  // simple and reliable.
  const processor = ctx.createScriptProcessor(4096, 1, 1)
  const chunks: Float32Array[] = []
  let stopRequested = false

  const cleanup = () => {
    try { processor.disconnect() } catch { /* noop */ }
    try { source.disconnect() } catch { /* noop */ }
    try { stream.getTracks().forEach(t => t.stop()) } catch { /* noop */ }
    void ctx.close().catch(() => {})
  }

  processor.onaudioprocess = (e) => {
    if (stopRequested) return
    const ch = e.inputBuffer.getChannelData(0)
    chunks.push(new Float32Array(ch))
  }
  source.connect(processor)
  processor.connect(ctx.destination)

  const t0 = performance.now()
  await new Promise<void>((resolve, reject) => {
    const tick = setInterval(() => {
      const elapsed = performance.now() - t0
      onProgress?.(elapsed)
      if (signal?.aborted) {
        clearInterval(tick)
        stopRequested = true
        cleanup()
        reject(new DOMException("Recording aborted", "AbortError"))
        return
      }
      if (elapsed >= seconds * 1000) {
        clearInterval(tick)
        stopRequested = true
        cleanup()
        resolve()
      }
    }, 100)
  })

  // Concatenate float32
  const total = chunks.reduce((s, c) => s + c.length, 0)
  const merged = new Float32Array(total)
  let off = 0
  for (const c of chunks) { merged.set(c, off); off += c.length }

  // Float32 [-1, 1] → Int16
  const i16 = new Int16Array(merged.length)
  for (let i = 0; i < merged.length; i++) {
    const x = Math.max(-1, Math.min(1, merged[i]))
    i16[i] = x < 0 ? x * 0x8000 : x * 0x7FFF
  }

  // RIFF/WAVE/PCM header (44 bytes)
  const sr = ctx.sampleRate
  const buffer = new ArrayBuffer(44 + i16.byteLength)
  const view = new DataView(buffer)
  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i))
  }
  writeStr(0, 'RIFF')
  view.setUint32(4, 36 + i16.byteLength, true)
  writeStr(8, 'WAVE')
  writeStr(12, 'fmt ')
  view.setUint32(16, 16, true)         // PCM chunk size
  view.setUint16(20, 1, true)          // format = PCM
  view.setUint16(22, 1, true)          // channels = 1
  view.setUint32(24, sr, true)         // sample rate
  view.setUint32(28, sr * 2, true)     // byte rate (sr * channels * bytes/sample)
  view.setUint16(32, 2, true)          // block align (channels * bytes/sample)
  view.setUint16(34, 16, true)         // bits per sample
  writeStr(36, 'data')
  view.setUint32(40, i16.byteLength, true)
  new Int16Array(buffer, 44).set(i16)
  return new Blob([buffer], { type: 'audio/wav' })
}

function VoiceProfilesCard() {
  const [profiles, setProfiles] = useState<Profile[]>([])
  const [loading, setLoading] = useState(true)
  const [name, setName] = useState('')
  const [recording, setRecording] = useState(false)
  const [recordElapsedMs, setRecordElapsedMs] = useState(0)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const r = await api.voiceSpeakersList()
      setProfiles(r.profiles || [])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const validName = /^[a-zA-Z0-9_-]+$/.test(name.trim())
  const trimmedName = name.trim()
  const nameTaken = profiles.some(p => p.name.toLowerCase() === trimmedName.toLowerCase())

  const startRecord = async () => {
    if (!trimmedName) {
      setError("Profile name required")
      return
    }
    if (!validName) {
      setError("Name must be alphanumeric (with - or _)")
      return
    }
    if (nameTaken) {
      setError(`'${trimmedName}' is already enrolled — delete it first or pick a different name`)
      return
    }
    setError(null)
    setSuccess(null)
    setRecording(true)
    setRecordElapsedMs(0)
    abortRef.current = new AbortController()
    try {
      const wav = await recordWav(
        RECORD_SECONDS,
        (ms) => setRecordElapsedMs(ms),
        abortRef.current.signal,
      )
      setRecording(false)
      setUploading(true)
      const r = await api.voiceSpeakersEnroll(trimmedName, wav)
      setSuccess(`Enrolled '${r.name}' (${r.duration_s}s @ ${r.sample_rate} Hz)`)
      setName('')
      await refresh()
    } catch (e) {
      if ((e as DOMException)?.name === 'AbortError') {
        // cancelled — silent
      } else {
        setError(e instanceof Error ? e.message : String(e))
      }
    } finally {
      setRecording(false)
      setUploading(false)
      abortRef.current = null
    }
  }

  const cancelRecord = () => {
    abortRef.current?.abort()
  }

  const deleteProfile = async (n: string) => {
    if (!confirm(`Delete voice profile '${n}'? This is irreversible.`)) return
    setError(null)
    setSuccess(null)
    try {
      await api.voiceSpeakersDelete(n)
      setSuccess(`Deleted '${n}'`)
      await refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const recordPct = Math.min(100, (recordElapsedMs / (RECORD_SECONDS * 1000)) * 100)

  return (
    <Card>
      <CardHeader>
        <CardTitle>Voice profiles</CardTitle>
        <CardDescription>
          When voice mode is on, every wake-word utterance is matched against enrolled profiles.
          A match prefixes the chat message with <code className="text-xs">[name]</code> and locks
          the continuation window to that speaker — utterances from a different voice during
          continuation are dropped.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Enroll form */}
        <div className="rounded-md border border-border bg-card/50 p-4 space-y-3">
          <div className="text-sm font-medium">Record a new profile</div>
          <div className="flex flex-col sm:flex-row gap-2">
            <Input
              placeholder="profile name (e.g. alan)"
              value={name}
              disabled={recording || uploading}
              onChange={e => setName(e.target.value)}
              className="sm:max-w-[280px]"
            />
            {recording ? (
              <Button variant="destructive" onClick={cancelRecord}>
                <Square className="w-4 h-4 mr-2" /> Cancel
              </Button>
            ) : (
              <Button
                onClick={startRecord}
                disabled={!trimmedName || !validName || nameTaken || uploading}
              >
                <Mic className="w-4 h-4 mr-2" />
                {uploading ? 'Uploading…' : `Record ${RECORD_SECONDS}s`}
              </Button>
            )}
          </div>

          {trimmedName && !validName && (
            <div className="text-xs text-destructive">
              Name must be alphanumeric (with - or _ allowed)
            </div>
          )}
          {trimmedName && validName && nameTaken && (
            <div className="text-xs text-amber-400">
              '{trimmedName}' is already enrolled
            </div>
          )}

          {recording && (
            <div className="space-y-1.5">
              <div className="text-xs text-muted-foreground">
                Recording — speak naturally for {RECORD_SECONDS} seconds…
                {' '}({(recordElapsedMs / 1000).toFixed(1)}s / {RECORD_SECONDS}s)
              </div>
              <div className="h-1.5 w-full bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-[width] duration-100"
                  style={{ width: `${recordPct}%` }}
                />
              </div>
            </div>
          )}

          <div className="text-xs text-muted-foreground">
            Tip: speak a sentence or two of regular conversation — "hi Lloyd, this is my
            voice, recording for the profile". Don't whisper or stay silent. The embedding
            is taken from this snippet, so what you say doesn't matter, but how you sound does.
          </div>
        </div>

        {/* Status */}
        {error && (
          <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span className="break-words">{error}</span>
          </div>
        )}
        {success && !error && (
          <div className="flex items-start gap-2 rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-400">
            <CheckCircle2 className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span className="break-words">{success}</span>
          </div>
        )}

        {/* Enrolled list */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm font-medium">
              Enrolled profiles
              {!loading && (
                <span className="ml-2 text-xs text-muted-foreground">({profiles.length})</span>
              )}
            </div>
            <Button variant="ghost" size="sm" onClick={refresh} disabled={loading}>
              <RefreshCw className={cn("w-3.5 h-3.5 mr-1.5", loading && "animate-spin")} /> Refresh
            </Button>
          </div>
          {loading ? (
            <div className="text-xs text-muted-foreground">Loading…</div>
          ) : profiles.length === 0 ? (
            <div className="rounded-md border border-dashed border-border px-4 py-6 text-center text-xs text-muted-foreground">
              No profiles enrolled yet. Record one above to get started.
            </div>
          ) : (
            <div className="space-y-1.5">
              {profiles.map(p => (
                <div
                  key={p.name}
                  className="flex items-center justify-between rounded-md border border-border bg-card/30 px-3 py-2"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <Mic className="w-4 h-4 text-primary flex-shrink-0" />
                    <span className="font-mono text-sm truncate">{p.name}</span>
                    <Badge variant="outline" className="text-[10px]">
                      {p.embedding_dim}-d
                    </Badge>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => deleteProfile(p.name)}
                    className="text-muted-foreground hover:text-destructive"
                    aria-label={`Delete ${p.name}`}
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

interface ClientCert {
  name: string
  fingerprint: string
  issued_at: string
}

interface NewlyMinted {
  name: string
  passphrase: string
}

function shortFp(fp: string): string {
  if (!fp) return ''
  const u = fp.toUpperCase()
  return `${u.slice(0, 8)}…${u.slice(-8)}`
}

function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

function DevicesCard() {
  const [info, setInfo] = useState<{
    lan_ip: string | null
    hostname: string
    https_url: string | null
    ca_available: boolean
  } | null>(null)
  const [identity, setIdentity] = useState<{ name: string | null; fingerprint: string | null } | null>(null)
  const [clients, setClients] = useState<ClientCert[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState<string | null>(null)
  const [newName, setNewName] = useState('')
  const [newPass, setNewPass] = useState('')
  const [minting, setMinting] = useState(false)
  const [justMinted, setJustMinted] = useState<NewlyMinted | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [i, id, cs] = await Promise.all([
        api.getLanInfo(),
        api.getIdentity(),
        api.listClients(),
      ])
      setInfo(i)
      setIdentity(id)
      setClients(cs.clients)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  const copy = async (label: string, value: string) => {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(label)
      setTimeout(() => setCopied(null), 1200)
    } catch {
      setError('Clipboard access denied')
    }
  }

  const downloadCa = async () => {
    setError(null)
    try {
      downloadBlob(await api.getCABlob(), 'lloyd-ca.crt')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const downloadP12 = async (name: string) => {
    setError(null)
    try {
      downloadBlob(await api.downloadClientP12(name), `lloyd-${name}.p12`)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const validName = /^[a-zA-Z0-9_-]+$/.test(newName.trim())
  const trimmedName = newName.trim()
  const nameTaken = clients.some(c => c.name.toLowerCase() === trimmedName.toLowerCase())

  const mint = async () => {
    if (!trimmedName || !validName || nameTaken) return
    setMinting(true)
    setError(null)
    setJustMinted(null)
    try {
      const r = await api.mintClient(trimmedName, newPass.trim() || undefined)
      setJustMinted({ name: r.name, passphrase: r.passphrase })
      setNewName('')
      setNewPass('')
      await refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setMinting(false)
    }
  }

  const revoke = async (name: string) => {
    const isYou = identity?.name === name
    const msg = isYou
      ? `'${name}' is the cert YOU are using. Revoking it will lock you out immediately. Continue?`
      : `Revoke client cert '${name}'? Devices using it will lose access immediately.`
    if (!confirm(msg)) return
    setError(null)
    try {
      await api.revokeClient(name)
      await refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Wifi className="w-5 h-5" /> LAN / remote access
        </CardTitle>
        <CardDescription>
          Lloyd uses mutual TLS — every device needs a client cert (signed by the on-host CA)
          to reach the API. Mint one cert per device and install it in that device's keystore.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {error && (
          <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
            <span className="break-words">{error}</span>
          </div>
        )}

        {/* URL + identity */}
        <div className="space-y-3">
          <div>
            <div className="text-xs uppercase tracking-wider text-muted-foreground mb-1">
              URL for other devices
            </div>
            {loading ? (
              <div className="text-sm text-muted-foreground">Detecting…</div>
            ) : info?.https_url ? (
              <div className="flex items-center gap-2">
                <code className="flex-1 rounded-md border border-border bg-card/50 px-3 py-2 text-sm font-mono break-all">
                  {info.https_url}
                </code>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => copy('url', info.https_url!)}
                  aria-label="Copy URL"
                >
                  {copied === 'url' ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                </Button>
              </div>
            ) : (
              <div className="text-sm text-amber-400">Could not detect a LAN IP.</div>
            )}
            {info?.hostname && (
              <div className="text-xs text-muted-foreground mt-1">
                Host: <span className="font-mono">{info.hostname}</span>
                {info.lan_ip && <> · IP: <span className="font-mono">{info.lan_ip}</span></>}
              </div>
            )}
          </div>

          {identity?.name && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              You are connected as <span className="font-mono text-foreground">{identity.name}</span>
              {identity.fingerprint && (
                <span className="text-muted-foreground">
                  · fp <span className="font-mono">{shortFp(identity.fingerprint)}</span>
                </span>
              )}
            </div>
          )}
        </div>

        {/* CA cert */}
        <div className="rounded-md border border-border bg-card/50 p-4 space-y-2">
          <div className="text-sm font-medium">Step 1 — install the CA on every device</div>
          <div className="text-xs text-muted-foreground">
            Each device must trust the Lloyd CA. Download once, install in the OS trust store.
            (Firefox keeps its own store — import via Settings → Privacy &amp; Security.)
          </div>
          <Button onClick={downloadCa} disabled={!info?.ca_available} variant="outline">
            <Download className="w-4 h-4 mr-2" /> Download CA cert (lloyd-ca.crt)
          </Button>
          {!info?.ca_available && (
            <div className="text-xs text-amber-400">
              CA not found. Run: <code>bash scripts/gen-cert.sh</code>
            </div>
          )}
        </div>

        {/* Mint device cert */}
        <div className="rounded-md border border-border bg-card/50 p-4 space-y-3">
          <div className="text-sm font-medium">Step 2 — mint a client cert for each device</div>
          <div className="text-xs text-muted-foreground">
            One cert per device. Install the .p12 in the device's OS keystore (or Firefox cert
            manager). The browser will offer the cert when connecting.
          </div>
          <div className="flex flex-col sm:flex-row gap-2">
            <Input
              placeholder="device name (e.g. phone, laptop)"
              value={newName}
              onChange={e => setNewName(e.target.value)}
              disabled={minting}
              className="sm:max-w-[220px]"
            />
            <Input
              placeholder="passphrase (default: lloyd)"
              value={newPass}
              onChange={e => setNewPass(e.target.value)}
              disabled={minting}
              className="sm:max-w-[220px]"
            />
            <Button
              onClick={mint}
              disabled={minting || !trimmedName || !validName || nameTaken}
            >
              <Plus className="w-4 h-4 mr-2" />
              {minting ? 'Minting…' : 'Mint + download'}
            </Button>
          </div>
          {trimmedName && !validName && (
            <div className="text-xs text-destructive">
              Name must be alphanumeric (with - or _ allowed)
            </div>
          )}
          {trimmedName && validName && nameTaken && (
            <div className="text-xs text-amber-400">
              '{trimmedName}' already exists — revoke it first or pick a different name
            </div>
          )}
          {justMinted && (
            <div className="rounded-md border border-emerald-500/40 bg-emerald-500/10 px-3 py-2 text-xs">
              <div className="flex items-start gap-2 text-emerald-300">
                <CheckCircle2 className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <div className="break-words space-y-1">
                  <div>Minted '<span className="font-mono">{justMinted.name}</span>'.
                    Passphrase: <span className="font-mono">{justMinted.passphrase}</span></div>
                  <Button
                    size="sm"
                    variant="outline"
                    className="mt-1"
                    onClick={() => downloadP12(justMinted.name)}
                  >
                    <Download className="w-3.5 h-3.5 mr-1.5" /> Download .p12
                  </Button>
                </div>
              </div>
            </div>
          )}

          <details className="text-xs text-muted-foreground">
            <summary className="cursor-pointer select-none hover:text-foreground">
              How to install a client cert per OS
            </summary>
            <div className="mt-2 space-y-2 pl-2 border-l-2 border-border">
              <div>
                <span className="font-medium text-foreground">macOS:</span> double-click the .p12
                → Keychain Access → enter passphrase → drag the imported cert to "login" or "System".
              </div>
              <div>
                <span className="font-medium text-foreground">Windows:</span> double-click → Certificate
                Import Wizard → enter passphrase → place in <em>Personal</em> store.
              </div>
              <div>
                <span className="font-medium text-foreground">Linux (Chrome):</span> Settings →
                Privacy &amp; Security → Security → Manage device certificates → "Your certificates"
                → Import.
              </div>
              <div>
                <span className="font-medium text-foreground">iOS:</span> AirDrop or email the .p12
                → Settings → "Profile Downloaded" → Install → enter passphrase. Then Settings →
                General → About → Certificate Trust Settings → enable for the Lloyd CA.
              </div>
              <div>
                <span className="font-medium text-foreground">Android:</span> Settings → Security →
                Encryption &amp; credentials → Install a certificate → VPN &amp; app user certificate.
              </div>
              <div>
                <span className="font-medium text-foreground">Firefox (any OS):</span> Settings →
                Privacy &amp; Security → Certificates → View Certificates → Your Certificates → Import.
              </div>
            </div>
          </details>
        </div>

        {/* Devices list */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm font-medium">
              Authorised devices
              {!loading && (
                <span className="ml-2 text-xs text-muted-foreground">({clients.length})</span>
              )}
            </div>
            <Button variant="ghost" size="sm" onClick={refresh} disabled={loading}>
              <RefreshCw className={cn("w-3.5 h-3.5 mr-1.5", loading && "animate-spin")} /> Refresh
            </Button>
          </div>
          {loading ? (
            <div className="text-xs text-muted-foreground">Loading…</div>
          ) : clients.length === 0 ? (
            <div className="rounded-md border border-dashed border-border px-4 py-6 text-center text-xs text-muted-foreground">
              No devices yet. Mint one above.
            </div>
          ) : (
            <div className="space-y-1.5">
              {clients.map(c => {
                const isYou = identity?.name === c.name
                return (
                  <div
                    key={c.name}
                    className="flex items-center justify-between rounded-md border border-border bg-card/30 px-3 py-2"
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <ShieldCheck className={cn("w-4 h-4 flex-shrink-0", isYou ? "text-emerald-400" : "text-muted-foreground")} />
                      <span className="font-mono text-sm truncate">{c.name}</span>
                      {isYou && <Badge variant="outline" className="text-[10px]">you</Badge>}
                      <span className="text-[10px] font-mono text-muted-foreground">
                        {shortFp(c.fingerprint)}
                      </span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => downloadP12(c.name)}
                        className="text-muted-foreground hover:text-foreground"
                        aria-label={`Download .p12 for ${c.name}`}
                      >
                        <Download className="w-4 h-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => revoke(c.name)}
                        className="text-muted-foreground hover:text-destructive"
                        aria-label={`Revoke ${c.name}`}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export default function SettingsPage() {
  return (
    <div className="p-6 max-w-3xl mx-auto space-y-6">
      <div>
        <h2 className="text-xl font-bold text-foreground">Settings</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Configure how Lloyd recognizes your voice and other preferences.
        </p>
      </div>
      <VoiceProfilesCard />
      <DevicesCard />
    </div>
  )
}
