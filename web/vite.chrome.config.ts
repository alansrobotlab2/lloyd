// Standalone Vite config for the Chrome side-panel extension.
//
// Builds two entries — `sidepanel.html` (React app in web/src/sidepanel)
// and `service-worker.ts` (in chrome-extension/src/background/) — and
// writes them, plus `manifest.json` and `icons/`, to
// chrome-extension/dist/.
//
// Usage:
//   cd web
//   VITE_API_BASE='http://127.0.0.1:8080/api' \
//     npx vite build -c vite.chrome.config.ts --watch
//
// The main app's `vite.config.ts` is unchanged.

import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"
import fs from "node:fs"
import path from "node:path"

const webDir = __dirname
const extDir = path.resolve(webDir, "../chrome-extension")
const extDist = path.join(extDir, "dist")

const manifestSrc = path.join(extDir, "manifest.json")

// The manifest is hand-written and copied, never generated (#1701). Check it
// before anything else: `emptyOutDir` below wipes dist/ before `closeBundle`
// runs, so a copy that fails there leaves an emptied dist/ behind and an
// ENOENT that names no path. The icons loop further down guards the same way.
function assertManifestPresent() {
  if (!fs.existsSync(manifestSrc)) {
    throw new Error(
      `chrome-extension/manifest.json is missing (looked for ${manifestSrc}); ` +
        "it is tracked since #1701 — restore it before building, dist/ was left untouched",
    )
  }
}
assertManifestPresent()

function copyExtensionStatic() {
  return {
    name: "lloyd-chrome-copy-static",
    closeBundle() {
      assertManifestPresent()
      fs.mkdirSync(extDist, { recursive: true })
      fs.copyFileSync(manifestSrc, path.join(extDist, "manifest.json"))
      const iconsSrc = path.join(extDir, "icons")
      if (fs.existsSync(iconsSrc)) {
        const iconsDst = path.join(extDist, "icons")
        fs.mkdirSync(iconsDst, { recursive: true })
        for (const f of fs.readdirSync(iconsSrc)) {
          fs.copyFileSync(path.join(iconsSrc, f), path.join(iconsDst, f))
        }
      }
    },
  }
}

export default defineConfig({
  plugins: [react(), tailwindcss(), copyExtensionStatic()],
  resolve: {
    alias: {
      "@": path.resolve(webDir, "./src"),
    },
  },
  build: {
    outDir: extDist,
    emptyOutDir: true,
    target: "chrome120",
    // Don't drag web/public/ (lloyd.jpg, favicon.svg) into the extension
    // bundle; icons + manifest are copied explicitly by the plugin below.
    copyPublicDir: false,
    rollupOptions: {
      input: {
        sidepanel: path.join(webDir, "sidepanel.html"),
        "service-worker": path.join(
          extDir,
          "src/background/service-worker.ts",
        ),
      },
      output: {
        entryFileNames: (info) =>
          info.name === "service-worker"
            ? "service-worker.js"
            : "assets/[name]-[hash].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
        format: "es",
      },
    },
  },
})
