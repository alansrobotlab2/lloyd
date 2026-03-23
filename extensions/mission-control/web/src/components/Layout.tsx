import React, { useState, useCallback, useEffect } from "react";
import Sidebar, { type Page } from "./Sidebar";
import ChatPanel from "./ChatPanel";
import { VoiceProvider } from "../contexts/VoiceContext";
import ServicesPage from "./pages/ServicesPage";
import DashboardPage from "./pages/DashboardPage";
import BacklogPage from "./pages/BacklogPage";
import MemoryPage from "./pages/MemoryPage";
import SkillsPage from "./pages/SkillsPage";
import SessionsPage from "./pages/SessionsPage";
import ModelsPage from "./pages/ModelsPage";
import CronPage from "./pages/CronPage";
import ToolsPage from "./pages/ToolsPage";
import AgentsPage from "./pages/AgentsPage";
import ActivityPage from "./pages/ActivityPage";
import SettingsPage from "./pages/SettingsPage";
import GraphPage from "./pages/GraphPage";
import ArchitecturePage from "./pages/ArchitecturePage";
import AutonomyPage from "./pages/AutonomyPage";

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div className="p-6 text-red-400">
          <h2 className="font-bold">Page Error</h2>
          <pre className="text-xs mt-2 whitespace-pre-wrap">{this.state.error.message}</pre>
          <button onClick={() => this.setState({ error: null })} className="mt-2 text-sm underline">
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const PAGES: Record<string, React.ComponentType> = {
  services: ServicesPage,
  dashboard: DashboardPage,
  backlog: BacklogPage,
  memory: MemoryPage,
  graph: GraphPage,
  skills: SkillsPage,
  models: ModelsPage,
  cron: CronPage,
  tools: ToolsPage,
  settings: SettingsPage,
  architecture: ArchitecturePage,
  autonomy: AutonomyPage,
};

export default function Layout() {
  const [page, setPage] = useState<Page>("chat");
  const [collapsed, setCollapsed] = useState(false);
  const [chatSessionKey, setChatSessionKey] = useState<string | null>(null);
  const [pendingAgentId, setPendingAgentId] = useState<string | null>(null);
  const [activeSessionKey, setActiveSessionKey] = useState<string | null>(null);

  // Auto-load session from URL query param ?session=<key>
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const sessionKey = params.get("session");
    if (sessionKey) {
      setChatSessionKey(sessionKey);
      setPage("chat");
    }
  }, []);

  const handleOpenSession = useCallback((sessionKey: string) => {
    setChatSessionKey(sessionKey);
    setPage("chat");
  }, []);

  const handleNavigateToAgent = useCallback((agentId: string) => {
    setPendingAgentId(agentId);
    setPage("agents");
  }, []);

  const handleSessionLoaded = useCallback(() => {
    setChatSessionKey(null);
  }, []);

  const PageComponent = PAGES[page];

  return (
    <ErrorBoundary>
      <VoiceProvider activeSessionKey={activeSessionKey}>
        <div className="h-screen flex bg-surface-0">
          <Sidebar
            active={page}
            onNavigate={setPage}
            collapsed={collapsed}
            onToggleCollapse={() => setCollapsed((c) => !c)}
            sessionKey={chatSessionKey || activeSessionKey}
          />
          <main className="flex-1 flex min-h-0 overflow-hidden">
            <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
              <div className={page === "chat" ? "flex flex-col min-h-0 flex-1 overflow-hidden" : "hidden"}>
                <ChatPanel requestedSessionKey={chatSessionKey} onSessionLoaded={handleSessionLoaded} onActiveSessionChange={setActiveSessionKey} />
              </div>
              {page === "sessions" && (
                <SessionsPage onOpenSession={handleOpenSession} />
              )}
              {page === "agents" && (
                <AgentsPage
                  initialAgentId={pendingAgentId}
                  onAgentIdConsumed={() => setPendingAgentId(null)}
                />
              )}
              {page === "activity" && (
                <ActivityPage onNavigateToAgent={handleNavigateToAgent} />
              )}
              {PageComponent && <PageComponent />}
            </div>
          </main>
        </div>
      </VoiceProvider>
    </ErrorBoundary>
  );
}
