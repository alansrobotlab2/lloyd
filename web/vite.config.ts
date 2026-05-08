import { defineConfig, type PluginOption } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import fs from "node:fs";
import path from "node:path";
import type { TLSSocket } from "node:tls";

const certDir = path.resolve(__dirname, "../agent-services/cert");
const serverCert = path.join(certDir, "lloyd.crt");
const serverKey = path.join(certDir, "lloyd.key");
const caCert = path.join(certDir, "ca.crt");

const haveCa = fs.existsSync(caCert);
const haveServer = fs.existsSync(serverCert) && fs.existsSync(serverKey);

if (!haveServer) {
  // eslint-disable-next-line no-console
  console.warn(
    "[vite] server cert missing — falling back to plain HTTP. Run: bash scripts/gen-cert.sh",
  );
}
if (haveServer && !haveCa) {
  // eslint-disable-next-line no-console
  console.warn(
    "[vite] CA cert missing — serving HTTPS without mTLS. Run: bash scripts/gen-cert.sh --force",
  );
}

const httpsConfig = haveServer
  ? {
      key: fs.readFileSync(serverKey),
      cert: fs.readFileSync(serverCert),
      ...(haveCa
        ? {
            ca: fs.readFileSync(caCert),
            requestCert: true,
            rejectUnauthorized: true,
          }
        : {}),
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
    },
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
