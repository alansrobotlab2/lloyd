/**
 * device-auth.ts — ED25519 device identity helpers for WebSocket authentication
 */

import { readFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import { createPrivateKey, createPublicKey, sign as cryptoSign } from "crypto";

const DEVICE_IDENTITY_PATH = join(homedir(), ".openclaw", "identity", "device.json");

const ED25519_SPKI_PREFIX = Buffer.from([0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00]);

export function base64UrlEncode(buf: Buffer): string {
  return buf.toString("base64").replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/g, "");
}

export function derivePublicKeyRaw(publicKeyPem: string): Buffer {
  const spki = createPublicKey(publicKeyPem).export({ type: "spki", format: "der" });
  if (spki.length === ED25519_SPKI_PREFIX.length + 32 && spki.subarray(0, ED25519_SPKI_PREFIX.length).equals(ED25519_SPKI_PREFIX))
    return spki.subarray(ED25519_SPKI_PREFIX.length);
  return spki;
}

export function loadDeviceIdentity(): { deviceId: string; publicKeyPem: string; privateKeyPem: string } | null {
  try {
    const raw = readFileSync(DEVICE_IDENTITY_PATH, "utf8");
    const parsed = JSON.parse(raw);
    if (parsed?.version === 1 && typeof parsed.deviceId === "string" && typeof parsed.publicKeyPem === "string" && typeof parsed.privateKeyPem === "string") {
      return { deviceId: parsed.deviceId, publicKeyPem: parsed.publicKeyPem, privateKeyPem: parsed.privateKeyPem };
    }
  } catch { /* non-fatal: identity file missing or malformed */ }
  return null;
}

export function signDevicePayload(privateKeyPem: string, payload: string): string {
  const key = createPrivateKey(privateKeyPem);
  return base64UrlEncode(cryptoSign(null, Buffer.from(payload, "utf8"), key));
}

export function buildDeviceAuthPayloadV3(params: {
  deviceId: string; clientId: string; clientMode: string; role: string;
  scopes: string[]; signedAtMs: number; token: string; nonce: string;
  platform: string; deviceFamily: string;
}): string {
  return ["v3", params.deviceId, params.clientId, params.clientMode, params.role,
    params.scopes.join(","), String(params.signedAtMs), params.token, params.nonce,
    params.platform, params.deviceFamily].join("|");
}
