import { Component, ErrorInfo, ReactNode } from 'react'
import Layout from './components/Layout'
import { VoiceModeProvider } from './contexts/VoiceModeContext'

interface ErrorBoundaryProps {
  children: ReactNode
}

interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-6 text-red-400">
          <h2 className="text-xl font-bold mb-2">Something went wrong</h2>
          <pre className="text-sm bg-slate-800 p-4 rounded overflow-auto">
            {this.state.error?.toString()}
          </pre>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="mt-4 px-4 py-2 bg-amber-500 text-slate-900 rounded"
          >
            Retry
          </button>
        </div>
      )
    }

    return this.props.children
  }
}

function App() {
  return (
    <ErrorBoundary>
      <VoiceModeProvider>
        <Layout />
      </VoiceModeProvider>
    </ErrorBoundary>
  )
}

export default App
