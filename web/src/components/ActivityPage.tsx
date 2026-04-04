export default function ActivityPage({ onNavigateToAgent }: { onNavigateToAgent: (agentId: string) => void }) {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  void onNavigateToAgent
  
  return (
    <div className="p-6">
      <h2 className="text-xl font-bold">Activity</h2>
      <p className="text-slate-400 mt-2">Activity log coming soon...</p>
    </div>
  )
}
