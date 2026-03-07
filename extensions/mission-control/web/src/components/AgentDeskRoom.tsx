/**
 * AgentDeskRoom — Visual "Agent Lab" showing active agents seated at desks
 * with animated avatars, thought bubbles, coffee mugs, and laptops.
 */
import { useState } from "react";
import { SubagentRunInfo, CcInstanceInfo } from "../api";
import type { SubagentDispatch } from "./pages/ActivityPage";

const MAX_DESKS = 8;

/** Unified desk occupant — from either legacy subagents or CC instances */
interface DeskOccupant {
  id: string;
  agentId: string;
  task: string;
  source: "subagent" | "cc";
}

interface Props {
  activeAgents: SubagentRunInfo[];
  ccInstances?: CcInstanceInfo[];
  activeDispatches?: SubagentDispatch[];
  onAgentClick?: (agentId: string) => void;
}

/** Extract agent id from legacy subagent label or childSessionKey */
function agentIdFromSubagent(agent: SubagentRunInfo): string | null {
  if (agent.label) {
    const id = agent.label.split(":")[0].trim().toLowerCase();
    if (id) return id;
  }
  if (agent.childSessionKey) {
    const parts = agent.childSessionKey.split(":");
    if (parts.length >= 2 && parts[0] === "agent") return parts[1];
  }
  return null;
}

/** Convert active subagents + CC running instances into unified desk occupants */
function buildOccupants(active: SubagentRunInfo[], ccInstances: CcInstanceInfo[], dispatches: SubagentDispatch[]): DeskOccupant[] {
  const occupants: DeskOccupant[] = [];

  // Legacy subagents
  for (const run of active) {
    const agentId = agentIdFromSubagent(run);
    if (agentId) {
      occupants.push({ id: run.runId, agentId, task: run.task, source: "subagent" });
    }
  }

  // CC instances — running only
  for (const inst of ccInstances) {
    if (inst.status !== "running") continue;
    const agentId = inst.type === "orchestrate" ? "orchestrator" : (inst.agent || "unknown");
    occupants.push({ id: inst.id, agentId, task: inst.activity || inst.task, source: "cc" });
  }

  // Orchestrator subagent dispatches (running only)
  for (const d of dispatches) {
    // Avoid duplicate if this agent already has a desk from CC instances
    if (occupants.some(o => o.agentId === d.agent)) continue;
    occupants.push({
      id: `dispatch-${d.agent}-${d.startTs}`,
      agentId: d.agent,
      task: `working...`,
      source: "cc",
    });
  }

  return occupants.slice(0, MAX_DESKS);
}

/** Truncate text for thought bubble */
function truncateStatus(text: string | undefined, max = 24): string {
  if (!text) return "";
  return text.length > max ? text.slice(0, max) + "\u2026" : text;
}

/** Split text into two lines at nearest word boundary */
function wrapLines(text: string): [string, string] {
  if (!text || text.length <= 12) return [text, ""];
  const mid = Math.floor(text.length / 2);
  let splitAt = -1;
  for (let i = 0; i <= mid; i++) {
    if (mid - i >= 0 && text[mid - i] === " ") { splitAt = mid - i; break; }
    if (mid + i < text.length && text[mid + i] === " ") { splitAt = mid + i; break; }
  }
  if (splitAt === -1) splitAt = mid;
  return [text.slice(0, splitAt).trim(), text.slice(splitAt).trim()];
}

function AvatarHead({ agentId, x, y, size = 36 }: { agentId: string | null; x: number; y: number; size?: number }) {
  const [failed, setFailed] = useState(false);

  if (!agentId || failed) {
    return (
      <g>
        <rect x={x} y={y} width={size} height={size} rx={size * 0.22} fill="#e8b88a">
          <animate attributeName="y" values={`${y};${y - 1};${y}`} dur="3s" repeatCount="indefinite" />
        </rect>
        <rect x={x + size * 0.22} y={y + size * 0.33} width={size * 0.17} height={size * 0.17} rx={size * 0.06} fill="#1e293b">
          <animate attributeName="height" values={`${size * 0.17};${size * 0.06};${size * 0.17}`} dur="4s" repeatCount="indefinite" />
        </rect>
        <rect x={x + size * 0.61} y={y + size * 0.33} width={size * 0.17} height={size * 0.17} rx={size * 0.06} fill="#1e293b">
          <animate attributeName="height" values={`${size * 0.17};${size * 0.06};${size * 0.17}`} dur="4s" repeatCount="indefinite" />
        </rect>
        <rect x={x + size * 0.33} y={y + size * 0.67} width={size * 0.33} height={size * 0.08} rx={size * 0.04} fill="#1e293b" opacity="0.5" />
      </g>
    );
  }

  return (
    <g>
      <defs>
        <clipPath id={`avatar-clip-${agentId}`}>
          <rect x={x} y={y} width={size} height={size} rx={size * 0.22} />
        </clipPath>
      </defs>
      <rect x={x - 1} y={y - 1} width={size + 2} height={size + 2} rx={size * 0.28} fill="rgba(99,102,241,0.3)">
        <animate attributeName="opacity" values="0.3;0.5;0.3" dur="3s" repeatCount="indefinite" />
      </rect>
      <image
        href={"/api/mc/agent-avatar?id=" + agentId}
        x={x}
        y={y}
        width={size}
        height={size}
        clipPath={`url(#avatar-clip-${agentId})`}
        onError={() => setFailed(true)}
      />
    </g>
  );
}

function DeskSlot({ occupant, index, onAgentClick }: { occupant?: DeskOccupant; index: number; onAgentClick?: (agentId: string) => void }) {
  const active = !!occupant;
  const agentId = occupant?.agentId ?? null;
  const DESK_MUG_COLORS: Record<string, string> = {
    coder:       "#7a9e87",  // sage green
    reviewer:    "#7a7aab",  // slate blue
    tester:      "#a87a7a",  // dusty rose
    planner:     "#7a8ea8",  // steel blue
    auditor:     "#9e7a87",  // dusty mauve
    operator:    "#7a9e9e",  // muted teal
    researcher:  "#a8927a",  // muted terracotta
    orchestrator:"#8a7aab",  // dusty purple
    clawhub:     "#9ea87a",  // olive green
    memory:      "#a8a07a",  // warm sand
  };
  const mugColor = agentId ? (DESK_MUG_COLORS[agentId] || "#8a8a9e") : "#8a8a9e";
  const statusText = truncateStatus(occupant?.task, 24);
  const [line1, line2] = wrapLines(statusText);
  const maxLineLen = Math.max(line1.length, line2?.length ?? 0);

  // Desk — brown, +50% wider, near bottom of 520x200 viewBox
  const deskTopY = 150;
  const deskFrontY = 162;
  const deskHeight = 16;
  const skew = 6;
  const cx = 85;
  const deskW = 165;
  const dL = cx - deskW / 2;
  const dR = cx + deskW / 2;

  // Character
  const avatarSize = 90;
  const avatarX = cx - avatarSize / 2;
  const avatarY = 24;
  const avatarCenterY = avatarY + avatarSize / 2;
  const bodyW = 36, bodyH = 24;
  const bodyX = cx - bodyW / 2;
  const bodyY = deskTopY - 48;
  const handW = 16, handH = 6;
  const handY = 130;

  // Thought bubble
  const bx = avatarX + avatarSize + 14;
  const bw = maxLineLen ? Math.max(280, maxLineLen * 22 + 56) : 0;
  const bh = line2 ? 116 : 68;
  const by = avatarCenterY - bh / 2;

  const clickable = active && !!agentId && !!onAgentClick;

  return (
    <div
      className={"desk-slot " + (active ? "desk-active" : "desk-empty") + (clickable ? " cursor-pointer" : "")}
      style={{ animationDelay: index * 0.12 + "s" }}
      onClick={clickable ? () => onAgentClick!(agentId!) : undefined}
      title={clickable ? `View ${agentId} agent` : undefined}
    >
      <svg viewBox="0 0 520 200" className="desk-svg" xmlns="http://www.w3.org/2000/svg">
        {/* Thought bubble */}
        {active && statusText && (
          <g className="thought-bubble" transform="matrix(1.0003523,0,0,0.81909953,-0.11201992,21.869887)">
            <circle cx={avatarX + avatarSize + 3} cy={avatarCenterY} r={6} fill="white" opacity="0.6">
              <animate attributeName="opacity" values="0.6;0.3;0.6" dur="2s" repeatCount="indefinite" />
            </circle>
            <circle cx={avatarX + avatarSize + 10} cy={avatarCenterY - 8} r={9} fill="white" opacity="0.7">
              <animate attributeName="opacity" values="0.7;0.4;0.7" dur="2s" begin="0.3s" repeatCount="indefinite" />
            </circle>
            <rect x={bx} y={by} width={bw} height={bh} rx={33} fill="#334155" stroke="#475569" strokeWidth="2" opacity="0.92" />
            <text x={bx + bw / 2} y={by + (line2 ? 45 : bh * 0.65)} textAnchor="middle" fontSize="22" fontFamily="monospace" fill="white" opacity="0.85">
              {line1}
            </text>
            {line2 && (
              <text x={bx + bw / 2} y={by + 85} textAnchor="middle" fontSize="22" fontFamily="monospace" fill="white" opacity="0.85">
                {line2}
              </text>
            )}
          </g>
        )}

        <g transform="matrix(1,0,0,1.0916577,0,-19.679914)">
        {/* Desk top face */}
        <polygon
          points={`${dL+skew},${deskTopY} ${dR+skew},${deskTopY} ${dR},${deskFrontY} ${dL},${deskFrontY}`}
          fill={active ? "#6b4226" : "#5a3720"}
          opacity={active ? 1 : 0.5}
        />
        <line x1={dL+skew} y1={deskTopY} x2={dR+skew} y2={deskTopY}
          stroke={active ? "#8b5e3c" : "#6b4226"} strokeWidth="1.5" opacity={active ? 0.8 : 0.4} />
        {/* Desk front face */}
        <polygon
          points={`${dL},${deskFrontY} ${dR},${deskFrontY} ${dR},${deskFrontY+deskHeight} ${dL},${deskFrontY+deskHeight}`}
          fill={active ? "#5a3720" : "#4a2e1a"}
          opacity={active ? 1 : 0.5}
        />
        <line x1={dL} y1={deskFrontY+deskHeight} x2={dR} y2={deskFrontY+deskHeight}
          stroke="#3d2416" strokeWidth="0.8" opacity="0.6" />
        {/* Desk legs */}
        <rect x={dL+4} y={deskFrontY+deskHeight} width="6" height="20" fill="#4a2e1a" opacity={active ? 0.8 : 0.4} />
        <rect x={dR-10} y={deskFrontY+deskHeight} width="6" height="20" fill="#4a2e1a" opacity={active ? 0.8 : 0.4} />
        <rect x={dL+4} y={deskFrontY+deskHeight+17} width={dR-dL-8} height="3" rx="1.5" fill="#3d2416" opacity={active ? 0.5 : 0.3} />

        {/* Coffee cup on left side of desk */}
        <g opacity={active ? 1 : 0.45} transform="translate(0, 10)">
          <polygon points={`14,${deskTopY} 30,${deskTopY} 29,${deskTopY-12} 15,${deskTopY-12}`} fill={mugColor} />
          <ellipse cx={22} cy={deskTopY-12} rx={8} ry={2.5} fill={mugColor} />
          <ellipse cx={22} cy={deskTopY-12} rx={6.5} ry={2} fill="#3d1f00" />
          <path d={`M30,${deskTopY-9} Q36,${deskTopY-9} 36,${deskTopY-5} Q36,${deskTopY-1} 30,${deskTopY-1}`} fill="none" stroke={mugColor} strokeWidth="2" />
          {active && (
            <g>
              <path d={`M19,${deskTopY-15} Q17,${deskTopY-20} 19,${deskTopY-25}`} stroke="rgba(200,200,200,0.6)" strokeWidth="1.5" fill="none" strokeLinecap="round">
                <animate attributeName="opacity" values="0;0.7;0" dur="2s" begin="0s" repeatCount="indefinite" />
                <animateTransform attributeName="transform" type="translate" values="0,0;-1,-6;-2,-12" dur="2s" begin="0s" repeatCount="indefinite" />
              </path>
              <path d={`M22,${deskTopY-15} Q24,${deskTopY-20} 22,${deskTopY-25}`} stroke="rgba(200,200,200,0.5)" strokeWidth="1.2" fill="none" strokeLinecap="round">
                <animate attributeName="opacity" values="0;0.6;0" dur="2.2s" begin="0.5s" repeatCount="indefinite" />
                <animateTransform attributeName="transform" type="translate" values="0,0;1,-7;0,-13" dur="2.2s" begin="0.5s" repeatCount="indefinite" />
              </path>
              <path d={`M25,${deskTopY-15} Q23,${deskTopY-20} 25,${deskTopY-25}`} stroke="rgba(200,200,200,0.4)" strokeWidth="1" fill="none" strokeLinecap="round">
                <animate attributeName="opacity" values="0;0.5;0" dur="1.8s" begin="1s" repeatCount="indefinite" />
                <animateTransform attributeName="transform" type="translate" values="0,0;0,-5;1,-11" dur="1.8s" begin="1s" repeatCount="indefinite" />
              </path>
            </g>
          )}
        </g>
        </g>

        {/* Laptop on desk */}
        {active ? (
          <g className="laptop" transform="translate(-2,-6)">
            <polygon
              points={`${cx-18+skew*0.5},${deskTopY+2} ${cx+18+skew*0.5},${deskTopY+2} ${cx+16+skew*0.3},${deskFrontY-2} ${cx-16+skew*0.3},${deskFrontY-2}`}
              fill="#1e293b" stroke="#334155" strokeWidth="0.8"
            />
            <polygon
              points={`${cx-17+skew*0.6},${deskTopY-26} ${cx+17+skew*0.6},${deskTopY-26} ${cx+18+skew*0.5},${deskTopY+2} ${cx-18+skew*0.5},${deskTopY+2}`}
              fill="#1a2332" stroke="#3b82f6" strokeWidth="1"
            >
              <animate attributeName="stroke-opacity" values="0.6;1;0.6" dur="3s" repeatCount="indefinite" />
            </polygon>
            <ellipse cx={cx+skew*0.3} cy={deskTopY-30} rx="24" ry="12" fill="#3b82f6" opacity="0.06">
              <animate attributeName="opacity" values="0.04;0.08;0.04" dur="3s" repeatCount="indefinite" />
            </ellipse>
            <ellipse cx={cx+skew*0.4} cy={deskTopY+4} rx="20" ry="5" fill="#3b82f6" opacity="0.08">
              <animate attributeName="opacity" values="0.05;0.1;0.05" dur="3s" repeatCount="indefinite" />
            </ellipse>
          </g>
        ) : (
          <g transform="translate(-2,-6)">
            <polygon
              points={`${cx-17+skew*0.6},${deskTopY-18} ${cx+17+skew*0.6},${deskTopY-18} ${cx+18+skew*0.5},${deskTopY+2} ${cx-18+skew*0.5},${deskTopY+2}`}
              fill="none" stroke="#5a3720" strokeWidth="0.5" opacity="0.3"
            />
          </g>
        )}

        {/* Character */}
        {active && (
          <g className="character-active">
            <rect x={bodyX} y={bodyY} width={bodyW} height={bodyH} rx={5} fill="#6366f1" />
            <AvatarHead agentId={agentId} x={avatarX} y={avatarY} size={avatarSize} />
            <rect x={54.77} y={handY} width={handW} height={handH} rx={3} fill="#e8b88a">
              <animate attributeName="y" values={`${handY};${handY+2};${handY}`} dur="0.3s" repeatCount="indefinite" />
            </rect>
            <rect x={101} y={handY} width={handW} height={handH} rx={3} fill="#e8b88a">
              <animate attributeName="y" values={`${handY+2};${handY};${handY+2}`} dur="0.3s" repeatCount="indefinite" />
            </rect>
          </g>
        )}

        {/* Agent name label */}
        {agentId && (
          <text x={512} y={194} textAnchor="end" fontSize="27" fontFamily="monospace" fill="white" opacity="0.7">
            {agentId}
          </text>
        )}
      </svg>
    </div>
  );
}

export default function AgentDeskRoom({ activeAgents, ccInstances, activeDispatches, onAgentClick }: Props) {
  const occupants = buildOccupants(activeAgents, ccInstances || [], activeDispatches || []);
  const activeCount = occupants.length;

  return (
    <div className="agent-desk-room">
      <div className="room-bg">
        <div className="room-wall" />
        <div className="room-floor" />
        <div className="room-particles">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="particle" style={{
              left: (15 + i * 14) + "%",
              animationDelay: (i * 1.2) + "s",
              animationDuration: (4 + (i % 3)) + "s",
            }} />
          ))}
        </div>
      </div>
      <div className="room-header">
        <span className="room-title" style={{ color: activeCount > 0 ? "#34d399" : "#64748b", transition: "color 0.6s ease" }}>Agent Lab</span>
        <span className="room-count">
          {activeCount > 0 ? (
            <><span className="count-dot" />{activeCount} working</>
          ) : (
            "all clear"
          )}
        </span>
      </div>
      <div className="desk-grid">
        {occupants.map((occ, i) => (
          <DeskSlot key={occ.id} occupant={occ} index={i} onAgentClick={onAgentClick} />
        ))}
        {/* Fill remaining empty desks */}
        {Array.from({ length: Math.max(0, MAX_DESKS - occupants.length) }).map((_, i) => (
          <DeskSlot key={`empty-${i}`} index={occupants.length + i} />
        ))}
      </div>
    </div>
  );
}
