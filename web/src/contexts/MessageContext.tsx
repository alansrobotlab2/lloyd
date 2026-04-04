import React, { createContext, useContext, useState, useCallback } from 'react'

export interface MessageEntry {
  id: string
  role: 'user' | 'assistant' | 'tool'
  content: Array<{ type: 'text'; text: string }>
  timestamp: string
  session_key?: string
}

export interface Session {
  id: string
  session_key: string
  display_name?: string
  preview?: string
  last_active: string
  platform?: string
}

interface MessageContextType {
  messages: MessageEntry[]
  addMessage: (role: 'user' | 'assistant' | 'tool', content: string) => void
  clearMessages: () => void
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
}

const MessageContext = createContext<MessageContextType | null>(null)

export function MessageProvider({ children }: { children: React.ReactNode }) {
  const [messages, setMessages] = useState<MessageEntry[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const addMessage = useCallback((role: 'user' | 'assistant' | 'tool', content: string) => {
    setMessages(prev => [
      ...prev,
      {
        id: `msg_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
        role,
        content: [{ type: 'text', text: content }],
        timestamp: new Date().toISOString(),
      },
    ])
  }, [])

  const clearMessages = useCallback(() => {
    setMessages([])
  }, [])

  return (
    <MessageContext.Provider value={{ messages, addMessage, clearMessages, isLoading, setIsLoading }}>
      {children}
    </MessageContext.Provider>
  )
}

export function useMessages() {
  const context = useContext(MessageContext)
  if (!context) {
    throw new Error('useMessages must be used within MessageProvider')
  }
  return context
}
