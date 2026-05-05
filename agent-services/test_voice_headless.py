#!/usr/bin/env python3
"""Headless voice pipeline test — no TUI, all output to stderr/stdout."""
import faulthandler
import os
import sys
import threading
import time
import signal

faulthandler.enable()

from voice_pipeline import load_config, PipelineRunner, State


class PrintCallbacks:
    def on_state_changed(self, state):
        print(f"[STATE] {state.name}", flush=True)

    def on_init_progress(self, component):
        print(f"[INIT] Loading: {component}", flush=True)

    def on_init_complete(self):
        print("[INIT] Complete — all models loaded", flush=True)
        print(f"[INFO] Active threads: {threading.active_count()}", flush=True)
        for t in threading.enumerate():
            print(f"  - {t.name} (daemon={t.daemon})", flush=True)

    def on_transcript(self, text, speaker, is_continuity):
        prefix = "[+] " if is_continuity else ""
        print(f"[TRANSCRIPT] {prefix}{speaker}: {text}", flush=True)

    def on_continuity_status(self, msg):
        print(f"[CONTINUITY] {msg}", flush=True)

    def on_error(self, error):
        print(f"[ERROR] {error}", flush=True)


def main():
    config = load_config("voice_bridge_config.json")
    runner = PipelineRunner(config, PrintCallbacks())

    print("--- Initializing components ---", flush=True)
    runner.init_components()

    # Report Speex state
    ww = runner.wake_word
    speex_status = "enabled" if ww.model.speex_ns else "disabled"
    print(f"[INFO] Speex noise suppression: {speex_status}", flush=True)

    print("--- Enabling voice mode ---", flush=True)
    runner.voice_enabled.set()
    runner.start()

    print("--- Voice mode active, press Ctrl+C to stop ---", flush=True)

    def on_sigint(sig, frame):
        print("\n--- Stopping ---", flush=True)
        runner.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_sigint)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
