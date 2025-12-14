'use client'

import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'

interface LogEntry {
  timestamp: string
  level: 'info' | 'success' | 'warning' | 'error'
  message: string
}

const initialLogs: LogEntry[] = [
  { timestamp: '18:42:17', level: 'info', message: '> alphaforge validate SPY --template sma_crossover --n-trials 1000 --run-cpcv --run-spa' },
  { timestamp: '18:42:18', level: 'info', message: 'Loading market data for SPY (2020-01-01 to 2024-12-13)...' },
  { timestamp: '18:42:19', level: 'success', message: '✓ Loaded 1,253 trading days' },
  { timestamp: '18:42:19', level: 'info', message: 'Generating strategy candidates via genetic programming...' },
  { timestamp: '18:42:47', level: 'success', message: '✓ Generated 10,000 candidates (population: 100, generations: 50)' },
  { timestamp: '18:42:47', level: 'info', message: 'Running DSR screen (threshold: 0.95)...' },
  { timestamp: '18:43:02', level: 'success', message: '✓ DSR Screen: 450/10,000 passed (4.5%)' },
  { timestamp: '18:43:02', level: 'info', message: 'Running CPCV validation (12,870 combinations)...' },
  { timestamp: '18:45:18', level: 'success', message: '✓ CPCV/PBO: 89/450 passed (PBO < 0.05)' },
  { timestamp: '18:45:18', level: 'info', message: 'Running event-driven backtests with queue models...' },
  { timestamp: '18:47:33', level: 'success', message: '✓ Event-Driven: 23/89 passed (shortfall < 30%)' },
  { timestamp: '18:47:33', level: 'info', message: 'Running Hansen SPA test vs SPY benchmark...' },
  { timestamp: '18:47:55', level: 'success', message: '✓ SPA Test: 8/23 passed (p < 0.05)' },
  { timestamp: '18:47:55', level: 'info', message: 'Running stress tests (6 scenarios)...' },
  { timestamp: '18:48:12', level: 'warning', message: '⚠ Stress: Crisis 2008 scenario failed (-43% DD)' },
  { timestamp: '18:48:12', level: 'success', message: '✓ Stress Tests: 5/6 passed (83%)' },
  { timestamp: '18:48:12', level: 'success', message: '━━━ VALIDATION COMPLETE ━━━' },
  { timestamp: '18:48:12', level: 'success', message: '✓ Strategy: SMA_Cross_Genetic_v47' },
  { timestamp: '18:48:12', level: 'success', message: '✓ Sharpe: 1.23, DSR: 0.96, PBO: 0.04' },
  { timestamp: '18:48:12', level: 'success', message: '✓ Recommendation: DEPLOY WITH APPROVAL' },
]

const levelColors = {
  info: 'text-terminal-text',
  success: 'text-terminal-green',
  warning: 'text-terminal-amber',
  error: 'text-terminal-red',
}

export default function CommandTerminal() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [command, setCommand] = useState('')
  const [showCursor, setShowCursor] = useState(true)

  useEffect(() => {
    // Animate logs appearing one by one
    initialLogs.forEach((log, idx) => {
      setTimeout(() => {
        setLogs(prev => [...prev, log])
      }, idx * 100)
    })

    // Cursor blink
    const cursorInterval = setInterval(() => {
      setShowCursor(prev => !prev)
    }, 500)

    return () => clearInterval(cursorInterval)
  }, [])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!command.trim()) return

    const newLog: LogEntry = {
      timestamp: new Date().toTimeString().split(' ')[0],
      level: 'info',
      message: `> ${command}`,
    }

    setLogs(prev => [...prev, newLog])

    // Simulate response
    setTimeout(() => {
      setLogs(prev => [...prev, {
        timestamp: new Date().toTimeString().split(' ')[0],
        level: 'info',
        message: 'Command received. Use the AlphaForge CLI to run actual validation.',
      }])
    }, 500)

    setCommand('')
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.8 }}
      className="border border-terminal-border bg-terminal-panel/50 backdrop-blur-sm rounded-lg p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="w-1 h-6 bg-terminal-blue" />
          <h2 className="text-xl font-display font-bold text-terminal-bright tracking-wide">
            SYSTEM TERMINAL
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-terminal-red rounded-full" />
          <div className="w-2 h-2 bg-terminal-amber rounded-full" />
          <div className="w-2 h-2 bg-terminal-green rounded-full animate-pulse" />
        </div>
      </div>

      {/* Terminal Output */}
      <div className="bg-terminal-bg border border-terminal-border rounded-lg p-4 h-80 overflow-y-auto font-mono text-xs">
        <div className="space-y-1">
          {logs.map((log, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.2 }}
              className={`${levelColors[log.level]} whitespace-pre-wrap`}
            >
              <span className="text-terminal-text/40">[{log.timestamp}]</span>{' '}
              {log.message}
            </motion.div>
          ))}
        </div>
      </div>

      {/* Command Input */}
      <form onSubmit={handleSubmit} className="mt-4">
        <div className="flex items-center gap-2 bg-terminal-bg border border-terminal-border rounded-lg p-3 font-mono text-sm">
          <span className="text-terminal-green">$</span>
          <input
            type="text"
            value={command}
            onChange={(e) => setCommand(e.target.value)}
            placeholder="alphaforge --help"
            className="flex-1 bg-transparent outline-none text-terminal-text placeholder:text-terminal-text/30"
          />
          {showCursor && <span className="w-2 h-4 bg-terminal-green" />}
        </div>
        <div className="mt-2 text-xs font-mono text-terminal-text/40">
          Try: <span className="text-terminal-amber">alphaforge validate SPY --template sma_crossover</span>
        </div>
      </form>
    </motion.div>
  )
}
