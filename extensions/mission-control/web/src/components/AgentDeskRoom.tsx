import { useState } from "react";
import { SubagentRunInfo } from "../api";

const MAX_DESKS = 8;

interface Props {
  activeAgents: SubagentRunInfo[];
}

/** Extract agent id from label — e.g. "coder", "memory" */
function agentIdFromLabel(agent?: SubagentRunInfo): string | null {
  if (!agent?.label) return null;
  return agent.label.split(":")[0].trim().toLowerCase() || null;
}

function AvatarHead({ agentId }: { agentId: string | null }) {
  const [failed, setFailed] = useState(false);

  if (!agentId || failed) {
    return (
      <g>
        <rect x="33" y="2" width="14" height="14" rx="3" fill="#e8b88a">
          <animate attributeName="y" values="2;1;2" dur="3s" repeatCount="indefinite" />
        </rect>
        <rect x="36" y="6" width="2" height="2" fill="#1e293b">
          <animate attributeName="height" values="2;0.5;2" dur="4s" repeatCount="indefinite" />
        </rect>
        <rect x="42" y="6" width="2" height="2" fill="#1e293b">
          <animate attributeName="height" values="2;0.5;2" dur="4s" repeatCount="indefinite" />
        </rect>
      </g>
    );
  }

  return (
    <g>
      <rect x="32" y="1" width="16" height="16" rx="4" fill="rgba(99,102,241,0.3)">
        <animate attributeName="opacity" values="0.3;0.5;0.3" dur="3s" repeatCount="indefinite" />
      </rect>
      <foreignObject x="33" y="2" width="14" height="14" style={{ overflow: "visible" }}>
        <img
          src={`/api/mc/agent-avatar?id=${agentId}`}
          alt={agentId}
          onError={() => setFailed(true)}
          style={{
            width: 14,
            height: 14,
            objectFit: "cover",
            borderRadius: 3,
            display: "block",
          }}
        />
      </foreignObject>
    </g>
  );
}

function DeskSlot({ agent, index }: { agent?: SubagentRunInfo; index: number }) {
  const active = !!agent;
  const label = agent?.label || agent?.task?.slice(0, 18) || "";
  const agentId = agentIdFromLabel(agent);

  return (
    <div
      className={`desk-slot ${active ? "desk-active" : "desk-empty"}`}
      style={{ animationDelay: `${index * 0.12}s` }}
    >
      <svg viewBox="0 0 80 72" className="desk-svg" xmlns="http://www.w3.org/2000/svg">
        {/* Monitor */}
        <rect x="24" y="8" width="32" height="22" rx="2"
          className={active ? "monitor-on" : "monitor-off"} />
        {active && (
          <g className="screen-content">
            <rect x="28" y="13" width="18" height="2" rx="1" fill="#4ade80" opacity="0.8">
              <animate attributeName="width" values="4;18;12;18" dur="2s" repeatCount="indefinite" />
            </rect>
            <rect x="28" y="17" width="12" height="2" rx="1" fill="#67e8f9" opacity="0.6">
              <animate attributeName="width" values="12;6;14;8" dur="2.4s" repeatCount="indefinite" />
            </rect>
            <rect x="28" y="21" width="8" height="2" rx="1" fill="#818cf8" opacity="0.5">
              <animate attributeName="width" values="8;16;4;10" dur="1.8s" repeatCount="indefinite" />
            </rect>
          </g>
        )}
        {active && (
          <rect x="24" y="8" width="32" height="22" rx="2" className="screen-glow" />
        )}
        {/* Monitor stand */}
        <rect x="37" y="30" width="6" height="4" fill="#334155" />
        <rect x="32" y="33" width="16" height="3" rx="1" fill="#293548" />
        {/* Desk surface */}
        <rect x="4" y="36" width="72" height="6" rx="2" fill="#3b4a5c" />
        <rect x="4" y="36" width="72" height="2" rx="1" fill="#475569" />
        {/* Desk legs */}
        <rect x="8" y="42" width="4" height="20" fill="#334155" />
        <rect x="68" y="42" width="4" height="20" fill="#334155" />
        <rect x="8" y="58" width="64" height="3" fill="#293548" />
        {/* Character */}
        {active ? (
          <g className="character-active">
            <AvatarHead agentId={agentId} />
            <rect x="34" y="16" width="12" height="10" rx="2" fill="#6366f1" />
            <rect x="26" y="28" width="8" height="3" rx="1" fill="#e8b88a">
              <animate attributeName="y" values="28;29;28" dur="0.3s" repeatCount="indefinite" />
            </rect>
            <rect x="46" y="28" width="8" height="3" rx="1" fill="#e8b88a">
              <animate attributeName="y" values="29;28;29" dur="0.3s" repeatCount="indefinite" />
            </rect>
          </g>
        ) : (
          <g className="character-absent">
            <rect x="34" y="22" width="12" height="10" rx="2" fill="#334155" opacity="0.3" />
          </g>
        )}
        <rect x="30" y="37" width="20" height="3" rx="1"
          fill={active ? "#475569" : "#334155"} opacity={active ? 1 : 0.4} />
      </svg>
      <div className={`desk-label ${active ? "desk-label-active" : "desk-label-empty"}`}>
        {active ? label : `Desk ${index + 1}`}
      </div>
      {active && <div className="desk-indicator" />}
    </div>
  );
}

export default function AgentDeskRoom({ activeAgents }: Props) {
  const slots: (SubagentRunInfo | undefined)[] = Array(MAX_DESKS).fill(undefined);
  activeAgents.slice(0, MAX_DESKS).forEach((agent, i) => {
    slots[i] = agent;
  });
  const activeCount = activeAgents.length;

  return (
    <div className="agent-desk-room">
      <div className="room-bg">
        <div className="room-wall" />
        <div className="room-floor" />
        <div className="room-particles">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="particle" style={{
              left: `${15 + i * 14}%`,
              animationDelay: `${i * 1.2}s`,
              animationDuration: `${4 + (i % 3)}s`,
            }} />
          ))}
        </div>
      </div>
      <div className="room-header">
        <span className="room-title">Agent Lab</span>
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
