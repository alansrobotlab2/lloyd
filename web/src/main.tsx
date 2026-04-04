import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

// Catch console errors for debugging
// eslint-disable-next-line @typescript-eslint/no-explicit-any
if ((window as any).process?.env?.NODE_ENV === 'development') {
  const originalError = console.error
  console.error = (...args) => {
    originalError.apply(console, args)
    if (args[0] && typeof args[0] === 'string' && args[0].includes('Error:')) {
      console.log('CAPTURED ERROR:', ...args)
    }
  }
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
