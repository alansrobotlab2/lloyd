import React, { useState, useCallback, useEffect } from "react";
import Sidebar, { type Page } from "./Sidebar";
import ChatPanel from "./ChatPanel";
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
};

export default function Layout() {
  const [page, setPage] = useState<Page>("chat");
  const [collapsed, setCollapsed] = useState(false);
  const [chatSessionId, setChatSessionId] = useState<string | null>(null);
  const [pendingAgentId, setPendingAgentId] = useState<string | null>(null);

  // Auto-load session from URL query param ?session=<id>
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const sessionId = params.get("session");
    if (sessionId) {
      setChatSessionId(sessionId);
      setPage("chat");
    }
  }, []);

  const handleOpenSession = useCallback((sessionId: string) => {
    setChatSessionId(sessionId);
    setPage("chat");
  }, []);

  const handleNavigateToAgent = useCallback((agentId: string) => {
    setPendingAgentId(agentId);
    setPage("agents");
  }, []);

  const handleSessionLoaded = useCallback(() => {
    setChatSessionId(null);
  }, []);

  const PageComponent = PAGES[page];

  return (
    <div className="h-screen flex bg-surface-0">
      <Sidebar
        active={page}
        onNavigate={setPage}
        collapsed={collapsed}
        onToggleCollapse={() => setCollapsed((c) => !c)}
      />
      <main className="flex-1 flex min-h-0 overflow-hidden">
        <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
          <ErrorBoundary>
            {page === "chat" && (
              <ChatPanel requestedSessionId={chatSessionId} onSessionLoaded={handleSessionLoaded} />
            )}
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
          </ErrorBoundary>
        </div>
      </main>
    </div>
  );
}
