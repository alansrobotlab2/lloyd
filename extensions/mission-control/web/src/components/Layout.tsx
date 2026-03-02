import { useState, useCallback } from "react";
import Sidebar, { type Page } from "./Sidebar";
import ChatPanel from "./ChatPanel";
import ServicesPage from "./pages/ServicesPage";
import DashboardPage from "./pages/DashboardPage";
import ClawDeckPage from "./pages/ClawDeckPage";
import MemoryPage from "./pages/MemoryPage";
import SkillsPage from "./pages/SkillsPage";
import SessionsPage from "./pages/SessionsPage";
import ModelsPage from "./pages/ModelsPage";
import CronPage from "./pages/CronPage";
import ToolsPage from "./pages/ToolsPage";
import AgentsPage from "./pages/AgentsPage";
import SettingsPage from "./pages/SettingsPage";

const PAGES: Record<string, React.ComponentType> = {
  services: ServicesPage,
  dashboard: DashboardPage,
  clawdeck: ClawDeckPage,
  memory: MemoryPage,
  skills: SkillsPage,
  agents: AgentsPage,
  models: ModelsPage,
  cron: CronPage,
  tools: ToolsPage,
  settings: SettingsPage,
};

export default function Layout() {
  const [page, setPage] = useState<Page>("chat");
  const [collapsed, setCollapsed] = useState(false);
  const [chatSessionId, setChatSessionId] = useState<string | null>(null);

  const handleOpenSession = useCallback((sessionId: string) => {
    setChatSessionId(sessionId);
    setPage("chat");
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
          {page === "chat" && (
            <ChatPanel requestedSessionId={chatSessionId} onSessionLoaded={handleSessionLoaded} />
          )}
          {page === "sessions" && (
            <SessionsPage onOpenSession={handleOpenSession} />
          )}
          {PageComponent && <PageComponent />}
        </div>
      </main>
    </div>
  );
}
