import { useState } from 'react'
import {
  LayoutList,
  Brain,
  Sparkles,
  Wrench,
  LayoutGrid,
  Settings,
  MessageCircle,
  Pin,
  PinOff,
  Code2,
  Lightbulb,
  Workflow,
  BrainCircuit,
  FileCode2,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'

export type Page = 'chat' | 'services' | 'backlog' | 'memory' | 'graph' | 'skills' | 'tools' | 'settings' | 'autonomy' | 'architecture' | 'workers' | 'inner_voice' | 'ide'

interface NavItem {
  id: Page
  label: string
  icon: React.ComponentType<{ className?: string }>
}

const NAV_ITEMS: NavItem[] = [
  { id: 'inner_voice', label: 'Inner Voice', icon: BrainCircuit },
  { id: 'chat', label: 'Chat', icon: MessageCircle },
  { id: 'backlog', label: 'Backlog', icon: LayoutGrid },
  { id: 'autonomy', label: 'Autonomy', icon: Lightbulb },
  { id: 'workers', label: 'Workers', icon: Workflow },
  { id: 'memory', label: 'Memory', icon: Brain },
  { id: 'architecture', label: 'Architecture', icon: Code2 },
  { id: 'ide', label: 'IDE', icon: FileCode2 },
  { id: 'skills', label: 'Skills', icon: Sparkles },
  { id: 'tools', label: 'Tools', icon: Wrench },
  { id: 'services', label: 'Services', icon: LayoutList },
]

const BOTTOM_ITEMS: NavItem[] = [
  { id: 'settings', label: 'Settings', icon: Settings },
]

interface SidebarProps {
  active: Page
  onNavigate: (page: Page) => void
  collapsed: boolean
  onToggleCollapse: () => void
  isMobile?: boolean
}

// Icon-button row with an optional tooltip that activates only when collapsed.
function NavRow({
  collapsed,
  label,
  showTooltip,
  className,
  onClick,
  disabled,
  children,
}: {
  collapsed: boolean
  label: string
  showTooltip?: boolean
  className?: string
  onClick?: () => void
  disabled?: boolean
  children: React.ReactNode
}) {
  const button = (
    <button
      onClick={onClick}
      disabled={disabled}
      className={cn(
        'w-full flex items-center rounded-md text-xs font-medium transition-colors',
        collapsed ? 'justify-center px-3 py-2' : 'gap-2.5 px-3 py-2',
        className,
      )}
    >
      {children}
    </button>
  )
  if (collapsed && showTooltip !== false) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>{button}</TooltipTrigger>
        <TooltipContent side="right">{label}</TooltipContent>
      </Tooltip>
    )
  }
  return button
}

export default function Sidebar({ active, onNavigate, collapsed, onToggleCollapse, isMobile }: SidebarProps) {
  // Hover-expand: when `collapsed` (the pinned state) is true, the sidebar
  // sits at 56px in-flow but pops out to 192px as an overlay on hover. When
  // `collapsed` is false, it's pinned-expanded and pushes content. The
  // collapse button toggles between those two modes.
  const [hovered, setHovered] = useState(false)
  const expanded = !collapsed || hovered
  const overlayMode = !isMobile && collapsed && hovered
  const PinIcon = collapsed ? PinOff : Pin

  const renderItem = (item: NavItem) => {
    const Icon = item.icon
    const isActive = active === item.id
    return (
      <NavRow
        key={item.id}
        collapsed={!expanded}
        label={item.label}
        onClick={() => onNavigate(item.id)}
        className={
          isActive
            ? 'bg-primary/15 text-primary hover:bg-primary/20'
            : 'text-muted-foreground hover:text-foreground hover:bg-accent'
        }
      >
        <Icon className="w-4 h-4 flex-shrink-0" />
        {expanded && <span className="truncate">{item.label}</span>}
      </NavRow>
    )
  }

  const aside = (
    <aside
      onMouseEnter={isMobile ? undefined : () => setHovered(true)}
      onMouseLeave={isMobile ? undefined : () => setHovered(false)}
      className={cn(
        'flex flex-col transition-all duration-200',
        isMobile ? 'w-full flex-1 py-2' : 'py-4 bg-card border-r border-border h-full',
        isMobile ? '' : expanded ? 'w-48' : 'w-14',
        // When hover-expanded over collapsed, lift out of the flow so we
        // overlay page content instead of shoving it.
        overlayMode && 'absolute inset-y-0 left-0 z-30 shadow-2xl shadow-black/40',
      )}
    >
      {/* Brand */}
      {!isMobile && (
        <div className={cn('mb-6 flex items-center gap-2', expanded ? 'px-4' : 'px-2 justify-center')}>
          <img src="/lloyd.jpg" alt="Lloyd" className="w-7 h-7 rounded-lg object-cover flex-shrink-0" />
          {expanded && (
            <div>
              <div className="text-sm font-bold tracking-wide text-foreground">LLOYD</div>
              <div className="text-[10px] text-muted-foreground -mt-0.5">Mission Control</div>
            </div>
          )}
        </div>
      )}

      {/* Main nav */}
      <nav className="flex-1 px-2 space-y-0.5">
        {NAV_ITEMS.map(renderItem)}
      </nav>

      {/* Bottom nav */}
      <div className="px-2 pt-2 space-y-0.5 border-t border-border">
        {BOTTOM_ITEMS.map(renderItem)}

        {!isMobile && (
          <>
            <Separator className="my-1" />
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onToggleCollapse}
                  className="w-full justify-center text-muted-foreground hover:text-foreground"
                >
                  <PinIcon className="w-4 h-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">
                {collapsed ? 'Pin sidebar open' : 'Unpin (auto-collapse)'}
              </TooltipContent>
            </Tooltip>
          </>
        )}
      </div>
    </aside>
  )

  // Mobile: rendered inside a Sheet, no positioning wrapper needed.
  if (isMobile) {
    return <TooltipProvider delayDuration={150}>{aside}</TooltipProvider>
  }

  // Desktop: outer div reserves the collapsed-width slot in the layout flow,
  // so when we promote the inner aside to `absolute` for the hover overlay,
  // page content doesn't reflow.
  return (
    <TooltipProvider delayDuration={150}>
      <div className={cn('relative shrink-0 transition-[width] duration-200', collapsed ? 'w-14' : 'w-48')}>
        {aside}
      </div>
    </TooltipProvider>
  )
}
