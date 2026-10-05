---
segment: memory
tags:
- memory
- daily-notes
type: note
---
## Self-mod guardian: supervisord was unreachable

supervisord was unreachable

Restarted agent-supervisord.service. No code was reverted — an unreachable supervisor is infrastructure, not a bad promotion.

--- evidence ---
- unreachable for 3 consecutive ticks (threshold 3, tick 5s)
- last probe error: <Fault 6: 'SHUTDOWN_STATE'>
- systemctl --user restart agent-supervisord.service: rc=0


## Self-mod guardian: supervisord was unreachable

supervisord was unreachable

Restarted agent-supervisord.service. No code was reverted — an unreachable supervisor is infrastructure, not a bad promotion.

--- evidence ---
- unreachable for 3 consecutive ticks (threshold 3, tick 5s)
- last probe error: <Fault 6: 'SHUTDOWN_STATE'>
- systemctl --user restart agent-supervisord.service: rc=0


## Self-mod guardian: supervisord was unreachable

supervisord was unreachable

Restarted agent-supervisord.service. No code was reverted — an unreachable supervisor is infrastructure, not a bad promotion.

--- evidence ---
- unreachable for 3 consecutive ticks (threshold 3, tick 5s)
- last probe error: [Errno 2] No such file or directory
- systemctl --user restart agent-supervisord.service: rc=0 stderr=Warning: The unit file, source configuration file or drop-ins of agent-supervisord.service changed on disk. Run 'systemctl --user daemon-reload' to reload units.


## Self-mod guardian: Service down, but no promotion to revert

Service down, but no promotion to revert

lloyd-mc:lloyd-backend: FATAL: can't find command '/home/alansrobotlab/lloyd/.venvs/lloyd/bin/python'

HEAD is 8328347d and no self-modification is being observed, so this is infrastructure rather than a bad change. Not rewriting history — this needs a human.


## Self-mod guardian: Runtime data is being written into the code tree

Runtime data is being written into the code tree

These exist inside the tree again:
  /home/alansrobotlab/lloyd/.t

Something still resolves a data path off the code instead of app.paths.DATA_ROOT (/home/alansrobotlab/lloyd-data). Find the writer, move the data across, and remove the in-tree copy.

If one of these is tooling or a rebuildable cache and not a writer, it belongs in KNOWN_GOOD_TOPLEVEL in agent-services/guardian/datawatch.py — adding its name there is what stops this alert; deleting the directory is not.

cleared: no runtime stores inside the code tree on the latest check — the instructions above are stale, nothing further to move

---

### Session 22:14 PDT — Auto-captured

The user requested research on OpenAI's Dots agents, prompting the assistant to clarify that Dots is a newly launched managed product within ChatGPT rather than a developer framework. The assistant explained that Dots are always-on personal agents powered by GPT-6 Astra with dedicated cloud resources, distinguishing them from the separate Agents API and SDK. Additionally, the assistant corrected a misconception by noting that Dots.llm1 is unrelated to OpenAI's offering.


## Self-mod guardian: Runtime data is being written into the code tree

Runtime data is being written into the code tree

These exist inside the tree again:
  /home/alansrobotlab/lloyd/.t

Something still resolves a data path off the code instead of app.paths.DATA_ROOT (/home/alansrobotlab/lloyd-data). Find the writer, move the data across, and remove the in-tree copy.

If one of these is tooling or a rebuildable cache and not a writer, it belongs in KNOWN_GOOD_TOPLEVEL in agent-services/guardian/datawatch.py — adding its name there is what stops this alert; deleting the directory is not.

cleared: no runtime stores inside the code tree on the latest check — the instructions above are stale, nothing further to move


## Self-mod guardian: Runtime data is being written into the code tree

Runtime data is being written into the code tree

These exist inside the tree again:
  /home/alansrobotlab/lloyd/.t

Something still resolves a data path off the code instead of app.paths.DATA_ROOT (/home/alansrobotlab/lloyd-data). Find the writer, move the data across, and remove the in-tree copy.

If one of these is tooling or a rebuildable cache and not a writer, it belongs in KNOWN_GOOD_TOPLEVEL in agent-services/guardian/datawatch.py — adding its name there is what stops this alert; deleting the directory is not.

cleared: no runtime stores inside the code tree on the latest check — the instructions above are stale, nothing further to move
