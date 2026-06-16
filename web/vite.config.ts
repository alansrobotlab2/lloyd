import { defineConfig, type PluginOption } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import importMetaUrlPlugin from "@codingame/esbuild-import-meta-url-plugin";
import fs from "node:fs";
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

/** Inject the peer cert's CN + sha256 fingerprint as request headers so the
 *  backend (FastAPI behind the /api proxy) can enforce a per-device allowlist
 *  at the HTTP layer. The TLS layer has already verified the cert was signed
 *  by our CA; this header is *trusted input* because Vite is the only path. */
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

export default defineConfig({
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
