import { useState } from "react";
import { SubagentRunInfo } from "../api";

const MAX_DESKS = 8;

interface Props {
  activeAgents: SubagentRunInfo[];
}

/** Extract agent id from label or childSessionKey — e.g. "coder", "memory" */
function agentIdFrom(agent?: SubagentRunInfo): string | null {
  if (!agent) return null;
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

/** Truncate text for thought bubble */
function truncateStatus(text: string | undefined, max = 20): string {
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

function ThoughtBubble({ text, x, y }: { text: string; x: number; y: number }) {
  const displayText = text || "...";
  const textWidth = Math.max(28, displayText.length * 5.2 + 12);
  const bubbleHeight = 16;

  const clampedY = Math.max(2, y);

  return (
    <g className="thought-bubble">
      <circle cx={x} cy={clampedY + 14} r="1.5" fill="white" opacity="0.7">
        <animate attributeName="opacity" values="0.7;0.3;0.7" dur="2s" repeatCount="indefinite" />
      </circle>
      <circle cx={x + 4} cy={clampedY + 9} r="2.5" fill="white" opacity="0.8">
        <animate attributeName="opacity" values="0.8;0.4;0.8" dur="2s" begin="0.3s" repeatCount="indefinite" />
      </circle>
      <rect
        x={x + 6}
        y={clampedY - 4}
        width={textWidth}
        height={bubbleHeight}
        rx="8"
        fill="white"
        stroke="#cbd5e1"
        strokeWidth="0.5"
        opacity="0.95"
      />
      <text
        x={x + 6 + textWidth / 2}
        y={clampedY + 7}
        textAnchor="middle"
        fontSize="7"
        fontFamily="monospace"
        fill="#1e293b"
        opacity="0.9"
      >
        {displayText}
      </text>
    </g>
  );
}

function DeskSlot({ agent, index }: { agent?: SubagentRunInfo; index: number }) {
  const active = !!agent;
  const agentId = agentIdFrom(agent);
  const statusText = truncateStatus(agent?.task, 24);
  const [line1, line2] = wrapLines(statusText);
  const maxLineLen = Math.max(line1.length, line2?.length ?? 0);

  // Desk — brown, +50% wider, near bottom of 520x200 viewBox
  const deskTopY = 150;
  const deskFrontY = 162;
  const deskHeight = 16;
  const skew = 6;
  const cx = 85;
  const deskW = 165; // 110 * 1.5
  const dL = cx - deskW / 2; // 2.5
  const dR = cx + deskW / 2; // 167.5

  // Character
  const avatarSize = 90;
  const avatarX = cx - avatarSize / 2; // 40
  const avatarY = 24;
  const avatarCenterY = avatarY + avatarSize / 2; // 69
  const bodyW = 36, bodyH = 24;
  const bodyX = cx - bodyW / 2; // 67
  const bodyY = deskTopY - 48; // 102
  const handW = 16, handH = 6;
  const handY = 130;

  // Thought bubble — 2x wider, 2x taller, fontSize +25%
  const bx = avatarX + avatarSize + 14; // 144
  const bw = maxLineLen ? Math.max(280, maxLineLen * 22 + 56) : 0;
  const bh = line2 ? 116 : 68;
  const by = avatarCenterY - bh / 2;

  return (
    <div
      className={"desk-slot " + (active ? "desk-active" : "desk-empty")}
      style={{ animationDelay: index * 0.12 + "s" }}
    >
      <svg viewBox="0 0 520 200" className="desk-svg" xmlns="http://www.w3.org/2000/svg">
        {/* Thought bubble — right of head, vertically centered, 2x size */}
        {active && statusText && (
          <g className="thought-bubble" transform="matrix(1.0003523,0,0,0.81909953,-0.11201992,21.869887)">
            <circle cx={avatarX + avatarSize + 3} cy={avatarCenterY} r={6} fill="white" opacity="0.6">
              <animate attributeName="opacity" values="0.6;0.3;0.6" dur="2s" repeatCount="indefinite" />
            </circle>
            <circle cx={avatarX + avatarSize + 10} cy={avatarCenterY - 8} r={9} fill="white" opacity="0.7">
              <animate attributeName="opacity" values="0.7;0.4;0.7" dur="2s" begin="0.3s" repeatCount="indefinite" />
            </circle>
            <rect x={bx} y={by} width={bw} height={bh} rx={33} fill="white" stroke="#cbd5e1" strokeWidth="2" opacity="0.92" />
            <text x={bx + bw / 2} y={by + (line2 ? 45 : bh * 0.65)} textAnchor="middle" fontSize="22" fontFamily="monospace" fill="#1e293b" opacity="0.85">
              {line1}
            </text>
            {line2 && (
              <text x={bx + bw / 2} y={by + 85} textAnchor="middle" fontSize="22" fontFamily="monospace" fill="#1e293b" opacity="0.85">
                {line2}
              </text>
            )}
          </g>
        )}

        <g transform="matrix(1,0,0,1.0916577,0,-19.679914)">
        {/* Desk top face — parallelogram (brown) */}
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
        </g>

        {/* Laptop on desk — centered on character */}
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
      </svg>
    </div>
  );
}

// DEBUG: all 8 slots active for animation testing
const DEBUG_AGENTS: SubagentRunInfo[] = [
  "coder", "planner", "researcher", "reviewer",
  "tester", "auditor", "operator", "memory",
].map((id) => ({
  runId: `debug-${id}`,
  childSessionKey: `agent:${id}:debug`,
  requesterSessionKey: "agent:main:debug",
  task: "working on something great",
  label: `${id}:debug`,
  createdAt: Date.now(),
}));

export default function AgentDeskRoom({ activeAgents }: Props) {
  const slots: (SubagentRunInfo | undefined)[] = activeAgents.slice(0, MAX_DESKS);
  const activeCount = activeAgents.length;

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
        {slots.map((agent, i) => (
          <DeskSlot key={i} agent={agent} index={i} />
        ))}
      </div>
    </div>
  );
}
