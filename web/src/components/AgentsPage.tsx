export default function AgentsPage({ initialAgentId, onAgentIdConsumed }: { initialAgentId: string | null, onAgentIdConsumed: () => void }) {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  void initialAgentId
  void onAgentIdConsumed
  
  return (
    <div className="p-6">
      <h2 className="text-xl font-bold">Agents</h2>
      <p className="text-slate-400 mt-2">Agent management coming soon...</p>
    </div>
  )
}
