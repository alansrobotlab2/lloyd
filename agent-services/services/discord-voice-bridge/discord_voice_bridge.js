#!/usr/bin/env node
/**
 * Discord Voice Bridge for Lloyd
 *
 * Connects to a Discord voice channel, captures per-user audio, resamples
 * 48 kHz stereo -> 16 kHz mono, and POSTs PCM frames to the Python bridge
 * server.  Polls for TTS audio and plays it back into the voice channel.
 */

"use strict";

const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");
const http = require("http");

const { Client, GatewayIntentBits } = require("discord.js");
const {
  joinVoiceChannel,
  createAudioPlayer,
  createAudioResource,
  AudioPlayerStatus,
  EndBehaviorType,
  VoiceConnectionStatus,
  entersState,
  StreamType,
} = require("@discordjs/voice");
const prism = require("prism-media");

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

function log(...args) {
  const ts = new Date().toISOString();
  console.log(`[${ts}]`, ...args);
}

function logError(...args) {
  const ts = new Date().toISOString();
  console.error(`[${ts}] ERROR:`, ...args);
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const CONFIG_PATH = path.join(__dirname, "config.json");
let config;
try {
  config = JSON.parse(fs.readFileSync(CONFIG_PATH, "utf8"));
} catch (err) {
  logError("Failed to read config.json:", err.message);
  process.exit(1);
}

const DISCORD_TOKEN = process.env.DISCORD_TOKEN || config.discord_token;
const GUILD_ID = config.guild_id;
const CHANNEL_ID = config.channel_id;
const BRIDGE_URL = config.bridge_url || "http://127.0.0.1:8096";

// ---------------------------------------------------------------------------
// Audio accumulator — collects PCM chunks and flushes every ~200ms
// ---------------------------------------------------------------------------

const FLUSH_SAMPLES = 3200; // 200ms at 16 kHz mono

class UserAudioAccumulator {
  constructor(userId, username) {
    this.userId = userId;
    this.username = username;
    this.chunks = [];
    this.totalSamples = 0;
  }

  /** Add a chunk; returns true if the buffer has reached the flush threshold. */
  addChunk(pcmInt16Mono16k) {
    this.chunks.push(pcmInt16Mono16k);
    this.totalSamples += pcmInt16Mono16k.length / 2; // int16 = 2 bytes per sample
    return this.totalSamples >= FLUSH_SAMPLES;
  }

  flush() {
    if (this.chunks.length === 0) return null;
    const combined = Buffer.concat(this.chunks);
    this.chunks = [];
    this.totalSamples = 0;
    return combined;
  }
}

// ---------------------------------------------------------------------------
// Resample 48 kHz stereo s16le -> 16 kHz mono s16le
// ---------------------------------------------------------------------------

function resample48kStereoTo16kMono(buf) {
  // Input: Int16 interleaved stereo at 48 kHz
  // Output: Int16 mono at 16 kHz
  const sampleCount = buf.length / 2; // total int16 samples (L+R interleaved)
  const framePairs = Math.floor(sampleCount / 2); // stereo frames
  // Decimate by 3: take every 3rd stereo frame, average L+R
  const outLen = Math.floor(framePairs / 3);
  const out = Buffer.alloc(outLen * 2);
  for (let i = 0; i < outLen; i++) {
    const srcFrame = i * 3; // stereo frame index
    const srcOffset = srcFrame * 4; // byte offset (2 channels * 2 bytes)
    const left = buf.readInt16LE(srcOffset);
    const right = buf.readInt16LE(srcOffset + 2);
    const mono = Math.round((left + right) / 2);
    out.writeInt16LE(Math.max(-32768, Math.min(32767, mono)), i * 2);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Send audio to bridge server
// ---------------------------------------------------------------------------

function sendAudioToBridge(userId, username, pcmBuffer) {
  const payload = JSON.stringify({
    user_id: userId,
    username: username,
    pcm_base64: pcmBuffer.toString("base64"),
    sample_rate: 16000,
  });

  const url = new URL(BRIDGE_URL + "/v1/discord_audio");
  const options = {
    hostname: url.hostname,
    port: url.port,
    path: url.pathname,
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Content-Length": Buffer.byteLength(payload),
    },
    timeout: 5000,
  };

  const req = http.request(options, (res) => {
    // Drain the response
    res.resume();
    if (res.statusCode !== 200) {
      logError(`Bridge returned ${res.statusCode} for audio POST`);
    }
  });
  req.on("error", (err) => {
    logError("Failed to send audio to bridge:", err.message);
  });
  req.write(payload);
  req.end();
}

// ---------------------------------------------------------------------------
// Poll for TTS audio from the bridge server
// ---------------------------------------------------------------------------

let player = null;
let connection = null;

function pollTtsQueue() {
  const url = new URL(BRIDGE_URL + "/v1/discord_tts_queue");
  const options = {
    hostname: url.hostname,
    port: url.port,
    path: url.pathname,
    method: "GET",
    timeout: 3000,
  };

  const req = http.request(options, (res) => {
    if (res.statusCode === 204) {
      // No content — nothing queued
      res.resume();
      return;
    }
    if (res.statusCode !== 200) {
      res.resume();
      return;
    }
    let body = "";
    res.on("data", (chunk) => { body += chunk; });
    res.on("end", () => {
      try {
        const data = JSON.parse(body);
        if (data.pcm_base64 && player && connection) {
          playTtsAudio(
            Buffer.from(data.pcm_base64, "base64"),
            data.sample_rate || 24000
          );
        }
      } catch (err) {
        logError("Failed to parse TTS queue response:", err.message);
      }
    });
  });
  req.on("error", () => {
    // Bridge may not be up yet — silently retry
  });
  req.end();
}

// ---------------------------------------------------------------------------
// Play TTS PCM audio into Discord
// ---------------------------------------------------------------------------

function playTtsAudio(pcmBuffer, sampleRate) {
  if (!player || !connection) return;

  log(`Playing TTS audio: ${pcmBuffer.length} bytes at ${sampleRate} Hz`);

  // Use ffmpeg to convert input PCM (mono, sampleRate) -> 48 kHz stereo s16le
  const ffmpeg = spawn("ffmpeg", [
    "-f", "s16le",
    "-ar", String(sampleRate),
    "-ac", "1",
    "-i", "pipe:0",
    "-ar", "48000",
    "-ac", "2",
    "-f", "s16le",
    "pipe:1",
  ], { stdio: ["pipe", "pipe", "ignore"] });

  ffmpeg.stdin.write(pcmBuffer);
  ffmpeg.stdin.end();

  const resource = createAudioResource(ffmpeg.stdout, {
    inputType: StreamType.Raw,
  });
  player.play(resource);
}

// ---------------------------------------------------------------------------
// Subscribe to a user's audio stream
// ---------------------------------------------------------------------------

function subscribeToUser(receiver, userId, username, activeSubscriptions) {
  log(`Subscribing to audio from ${username} (${userId})`);

  const opusStream = receiver.subscribe(userId, {
    end: {
      behavior: EndBehaviorType.AfterSilence,
      duration: 1000,
    },
  });

  // Decode opus -> PCM 48 kHz stereo s16le
  const decoder = new prism.opus.Decoder({
    rate: 48000,
    channels: 2,
    frameSize: 960,
  });

  const accumulator = new UserAudioAccumulator(userId, username);

  opusStream.pipe(decoder);

  decoder.on("data", (pcmChunk) => {
    // pcmChunk is 48 kHz stereo s16le — resample to 16 kHz mono
    const mono16k = resample48kStereoTo16kMono(pcmChunk);
    const ready = accumulator.addChunk(mono16k);
    // Flush to bridge in real-time every ~200ms of audio
    if (ready) {
      const pcmBuffer = accumulator.flush();
      if (pcmBuffer && pcmBuffer.length > 640) {
        sendAudioToBridge(userId, username, pcmBuffer);
      }
    }
  });

  decoder.on("end", () => {
    // Stream ended (silence detected) — flush remaining audio and clean up
    const pcmBuffer = accumulator.flush();
    if (pcmBuffer && pcmBuffer.length > 640) {
      log(`Flushing final ${pcmBuffer.length} bytes from ${username} to bridge`);
      sendAudioToBridge(userId, username, pcmBuffer);
    }
    activeSubscriptions.delete(userId);
  });

  decoder.on("error", (err) => {
    logError(`Decoder error for ${username}:`, err.message);
    activeSubscriptions.delete(userId);
  });

  opusStream.on("error", (err) => {
    logError(`Opus stream error for ${username}:`, err.message);
  });
}

// ---------------------------------------------------------------------------
// Main — connect to Discord and join voice channel
// ---------------------------------------------------------------------------

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildVoiceStates,
  ],
});

client.once("ready", async () => {
  log(`Logged in as ${client.user.tag}`);

  const guild = client.guilds.cache.get(GUILD_ID);
  if (!guild) {
    logError(`Guild ${GUILD_ID} not found`);
    process.exit(1);
  }

  const channel = guild.channels.cache.get(CHANNEL_ID);
  if (!channel) {
    logError(`Channel ${CHANNEL_ID} not found in guild ${GUILD_ID}`);
    process.exit(1);
  }

  log(`Joining voice channel: ${channel.name} (${CHANNEL_ID})`);

  // Create audio player
  player = createAudioPlayer();
  player.on("error", (err) => {
    logError("Audio player error:", err.message);
  });
  player.on(AudioPlayerStatus.Idle, () => {
    // Ready for next TTS audio
  });

  // Join the voice channel
  connection = joinVoiceChannel({
    channelId: CHANNEL_ID,
    guildId: GUILD_ID,
    adapterCreator: guild.voiceAdapterCreator,
    selfDeaf: false,
    selfMute: false,
  });

  // Subscribe player to connection
  connection.subscribe(player);

  // Handle connection state changes
  connection.on(VoiceConnectionStatus.Ready, () => {
    log("Voice connection ready");
  });

  connection.on(VoiceConnectionStatus.Disconnected, async () => {
    log("Voice connection disconnected — attempting reconnect...");
    try {
      await Promise.race([
        entersState(connection, VoiceConnectionStatus.Signalling, 5000),
        entersState(connection, VoiceConnectionStatus.Connecting, 5000),
      ]);
      // Reconnecting automatically
      log("Reconnecting...");
    } catch {
      // Connection is truly dead — rejoin
      log("Reconnect failed — rejoining channel...");
      connection.destroy();
      connection = joinVoiceChannel({
        channelId: CHANNEL_ID,
        guildId: GUILD_ID,
        adapterCreator: guild.voiceAdapterCreator,
        selfDeaf: false,
        selfMute: false,
      });
      connection.subscribe(player);
    }
  });

  // Wait for the connection to be ready
  try {
    await entersState(connection, VoiceConnectionStatus.Ready, 20000);
  } catch {
    logError("Failed to join voice channel within 20s");
    process.exit(1);
  }

  // Listen for users speaking
  const receiver = connection.receiver;
  const activeSubscriptions = new Set();

  receiver.speaking.on("start", (userId) => {
    if (activeSubscriptions.has(userId)) return;
    activeSubscriptions.add(userId);

    // Look up the username
    const member = guild.members.cache.get(userId);
    const username = member ? member.displayName : `User-${userId}`;

    subscribeToUser(receiver, userId, username, activeSubscriptions);
  });

  // Start polling for TTS audio
  setInterval(pollTtsQueue, 500);

  log("Discord voice bridge is running");
});

// Graceful shutdown
process.on("SIGINT", () => {
  log("Shutting down...");
  if (connection) connection.destroy();
  client.destroy();
  process.exit(0);
});

process.on("SIGTERM", () => {
  log("Received SIGTERM, shutting down...");
  if (connection) connection.destroy();
  client.destroy();
  process.exit(0);
});

// Login
client.login(DISCORD_TOKEN).catch((err) => {
  logError("Failed to login to Discord:", err.message);
  process.exit(1);
});
