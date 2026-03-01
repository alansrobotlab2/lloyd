import StatCards from "../StatCards";
import UsageChart from "../UsageChart";
import ApiCallsTable from "../ApiCallsTable";

export default function DashboardPage() {
  return (
    <div className="h-full flex flex-col gap-4 p-4 overflow-auto">
      <StatCards />
      <UsageChart />
      <div className="flex-1 min-h-[200px]">
        <ApiCallsTable />
      </div>
    </div>
  );
}
