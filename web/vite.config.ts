import { defineConfig, type PluginOption } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import importMetaUrlPlugin from "@codingame/esbuild-import-meta-url-plugin";
import { createHash } from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import type { TLSSocket } from "node:tls";

const certDir = path.resolve(__dirname, "../agent-services/cert");

// Prefer the publicly-trusted Tailscale (Let's Encrypt) cert for the tailnet
// MagicDNS name if it's been provisioned (`tailscale cert <name>`). It's
// trusted by every device out of the box — no CA install, no warnings, works
// in standalone home-screen apps. Falls back to the private Lloyd server cert.
const tsHost = "goliath.taile37041.ts.net";
const tsCert = path.join(certDir, `${tsHost}.crt`);
const tsKey = path.join(certDir, `${tsHost}.key`);
const haveTs = fs.existsSync(tsCert) && fs.existsSync(tsKey);

const serverCert = haveTs ? tsCert : path.join(certDir, "lloyd.crt");
const serverKey = haveTs ? tsKey : path.join(certDir, "lloyd.key");

const haveServer = fs.existsSync(serverCert) && fs.existsSync(serverKey);

// eslint-disable-next-line no-console
console.log(
  haveTs
    ? `[vite] HTTPS using Tailscale public cert for ${tsHost}`
    : "[vite] HTTPS using private Lloyd cert (no Tailscale cert found yet)",
);
if (!haveServer) {
  // eslint-disable-next-line no-console
  console.warn(
    "[vite] server cert missing — falling back to plain HTTP. Run: bash scripts/gen-cert.sh",
  );
}

// mTLS dropped 2026-06-14: client-cert auth can't work in iOS Chrome (and
// other third-party iOS browsers) — they can't present keychain identities
// for mutual TLS, only Safari can. Tailscale is the access boundary now:
// only tailnet devices can reach :5173. We still serve HTTPS with the Lloyd
// server cert (encrypted + secure context for SSE/voice); we just no longer
// request or require a client cert, so any browser on the tailnet works.
const httpsConfig = haveServer
  ? {
      key: fs.readFileSync(serverKey),
      cert: fs.readFileSync(serverCert),
    }
  : undefined;

/** Inert since mTLS was dropped (5e1351f3): httpsConfig above carries no
 *  requestCert/ca, so no browser presents a client cert, getPeerCertificate()
 *  has no subject, and this plugin sets neither x-client-cn nor
 *  x-client-fingerprint. Kept wired so a cert-bearing client still gets its
 *  fingerprint forwarded.
 *
 *  Neither header is an authentication input. x-client-fingerprint is read
 *  only by server._cert_fingerprint, strictly after ApiPeerGate's peer-address
 *  refusal, and is deny-only: a value absent from
 *  agent-services/cert/clients.json produces a 403 and no value ever grants
 *  access. That allowlist is empty, so a request that does send a fingerprint
 *  is refused. x-client-cn has no server-side reader. */
function clientCertHeaders(): PluginOption {
  return {
    name: "lloyd-client-cert-headers",
    configureServer(server) {
      server.middlewares.use((req, _res, next) => {
        const sock = req.socket as TLSSocket
        if (typeof sock?.getPeerCertificate === "function") {
          const cert = sock.getPeerCertificate()
          if (cert && cert.subject) {
            const cn = (cert.subject as { CN?: string }).CN || ""
            const fp = (cert.fingerprint256 || "").replace(/:/g, "")
            if (cn) req.headers["x-client-cn"] = cn
            if (fp) req.headers["x-client-fingerprint"] = fp
          }
        }
        next()
      })
    },
  }
}

// A checkout that borrows the live tree's `node_modules` through a symlink
// (a round worktree — `gate.py`'s frontend rung makes the link — or the
// sandbox) must not share its dep cache too: the default cacheDir is
// `node_modules/.vite`, so a second dev server there re-optimizes over the
// live server's chunks, and the live server keeps handing out hashed chunk
// names that are gone ("Failed to fetch dynamically imported module", a 504
// on :5173, 2026-10-03). Such a checkout gets a cache of its own, keyed by
// its path; the tree that owns `node_modules` keeps the default.
function ownCacheDir(): string | undefined {
  const nm = path.resolve(__dirname, "node_modules");
  let borrowed = false;
  try {
    borrowed = fs.lstatSync(nm).isSymbolicLink();
  } catch {
    return undefined;
  }
  if (!borrowed) return undefined;
  const key = createHash("sha1").update(__dirname).digest("hex").slice(0, 12);
  return path.join(os.tmpdir(), `lloyd-vite-cache-${key}`);
}

export default defineConfig({
  cacheDir: ownCacheDir(),
  plugins: [clientCertHeaders(), react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      // Route every `monaco-editor` import (ours, @monaco-editor/react's,
      // monaco-languageclient's, etc.) to the codingame VSCode-flavored
      // build. Without a single shared Monaco runtime, language client
      // provider registrations don't reach our editor instances.
      "monaco-editor": "@codingame/monaco-vscode-editor-api",
    },
  },
  // The codingame packages use `new Worker(new URL(..., import.meta.url))`
  // to load Monaco's editor / textmate / extension-host workers. Vite's
  // dep pre-bundler rewrites `import.meta.url` to point at the optimized
  // chunk, which breaks those worker URLs (browser ends up fetching
  // index.html instead of the worker JS). The esbuild plugin below
  // preserves the original URLs during pre-bundling so workers and
  // bundled extension assets resolve correctly.
  optimizeDeps: {
    esbuildOptions: {
      plugins: [importMetaUrlPlugin as unknown as never],
    },
    // The codingame packages and monaco-languageclient must NOT be
    // pre-bundled — they use `new URL(..., import.meta.url)` for workers
    // and rely on side-effect imports (vscode/localExtensionHost) that
    // Vite's optimizer would break.
    //
    // BUT: vscode-languageclient is pure CJS that uses `__exportStar`
    // runtime re-exports for BaseLanguageClient. It HAS to be pre-bundled
    // by esbuild so the browser sees named ESM exports. Same for
    // vscode-languageserver-protocol (ditto) and the cmdk transitive deps.
    exclude: [
      "monaco-languageclient",
      "monaco-languageclient/vscodeApiWrapper",
      "monaco-languageclient/workerFactory",
      "monaco-languageclient/wrapper",
      "monaco-languageclient/editorApp",
      "@codingame/monaco-vscode-api",
      "@codingame/monaco-vscode-editor-api",
      "vscode",
    ],
    // Force pre-bundle for the LSP CJS deps so their __exportStar named
    // exports get materialised into proper ESM by esbuild.
    include: [
      "vscode-languageclient/browser.js",
      "vscode-languageserver-protocol",
      "vscode-jsonrpc",
    ],
    needsInterop: [
      "vscode-languageclient/browser.js",
      "vscode-languageserver-protocol",
      "vscode-jsonrpc",
    ],
  },
  worker: {
    format: "es",
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    https: httpsConfig,
    allowedHosts: true,
    proxy: {
      "/api": {
        target: "http://localhost:8080",
        changeOrigin: true,
        xfwd: true,
        // ws: true is required for WebSocket upgrades on /api/* — without
        // this flag, the LSP WebSocket endpoints (/api/lsp/{language}) die
        // at the Vite layer and never reach the backend, which silently
        // breaks all language-server features (hover, go-to-def, etc).
        ws: true,
        timeout: 300000,
      },
      "/livekit": {
        target: "ws://127.0.0.1:7880",
        ws: true,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/livekit/, ""),
      },
    },
  },
});
