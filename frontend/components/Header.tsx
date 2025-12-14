'use client'

import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'

export default function Header() {
  const [time, setTime] = useState(new Date())

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  return (
    <motion.header
      initial={{ opacity: 0, y: -50 }}
      animate={{ opacity: 1, y: 0 }}
      className="border-b border-terminal-border bg-terminal-panel/80 backdrop-blur-sm sticky top-0 z-50"
    >
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo and Status */}
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-3">
              <svg className="w-8 h-8 text-terminal-green" viewBox="0 0 24 24" fill="none">
                <path d="M3 3L21 3L21 21L3 21L3 3Z" stroke="currentColor" strokeWidth="2"/>
                <path d="M7 12L12 7L17 12L12 17L7 12Z" fill="currentColor"/>
                <circle cx="12" cy="12" r="2" fill="#0a0e14"/>
              </svg>
              <span className="text-2xl font-display font-bold text-terminal-bright tracking-wider">
                ALPHAFORGE
              </span>
            </div>

            <div className="hidden md:flex items-center gap-2 text-xs font-mono">
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 bg-terminal-green rounded-full animate-pulse" />
                <span className="text-terminal-green">ACTIVE</span>
              </div>
              <span className="text-terminal-text/40">|</span>
              <span className="text-terminal-text/60">MVP7 Complete</span>
              <span className="text-terminal-text/40">|</span>
              <span className="text-terminal-text/60">300+ Tests Passing</span>
            </div>
          </div>

          {/* System Time and Stats */}
          <div className="flex items-center gap-6">
            <div className="hidden lg:flex items-center gap-4 text-xs font-mono text-terminal-text/60">
              <div>
                <span className="text-terminal-text/40">UTC:</span>{' '}
                <span className="text-terminal-amber metric-value">
                  {time.toUTCString().split(' ')[4]}
                </span>
              </div>
              <div>
                <span className="text-terminal-text/40">SYS:</span>{' '}
                <span className="text-terminal-green metric-value">98.3%</span>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-2">
              <button className="px-4 py-2 bg-terminal-panel border border-terminal-border rounded text-xs font-mono text-terminal-text hover:border-terminal-green hover:text-terminal-green transition-all">
                VALIDATE
              </button>
              <button className="px-4 py-2 bg-terminal-green text-terminal-bg rounded text-xs font-mono font-semibold hover:bg-terminal-green/90 transition-all">
                NEW STRATEGY
              </button>
            </div>
          </div>
        </div>
      </div>
    </motion.header>
  )
}
