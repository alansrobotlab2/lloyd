export default function SessionsPage({ onOpenSession }: { onOpenSession: (sessionKey: string) => void }) {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  void onOpenSession
  
  return (
    <div className="p-6">
      <h2 className="text-xl font-bold">Sessions</h2>
      <p className="text-slate-400 mt-2">Session management coming soon...</p>
    </div>
  )
}
