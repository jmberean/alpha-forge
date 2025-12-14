'use client'

import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'
import { listStrategies, Strategy } from '@/lib/api'

interface DisplayStrategy extends Strategy {
  pbo?: number
  returns?: number
}

const statusConfig = {
  deployed: { color: 'text-terminal-green', bg: 'bg-terminal-green/10', border: 'border-terminal-green', label: 'LIVE' },
  approved: { color: 'text-terminal-amber', bg: 'bg-terminal-amber/10', border: 'border-terminal-amber', label: 'APPROVED' },
  passed: { color: 'text-terminal-green', bg: 'bg-terminal-green/10', border: 'border-terminal-green', label: 'PASSED' },
  testing: { color: 'text-terminal-blue', bg: 'bg-terminal-blue/10', border: 'border-terminal-blue', label: 'TESTING' },
  rejected: { color: 'text-terminal-red', bg: 'bg-terminal-red/10', border: 'border-terminal-red', label: 'FAILED' },
  failed: { color: 'text-terminal-red', bg: 'bg-terminal-red/10', border: 'border-terminal-red', label: 'FAILED' },
}

export default function StrategyList() {
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null)
  const [strategies, setStrategies] = useState<DisplayStrategy[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchStrategies = async () => {
      try {
        setLoading(true)
        const data = await listStrategies()
        setStrategies(data.map(s => ({
          ...s,
          returns: s.annual_return * 100,
        })))
        setError(null)
      } catch (err) {
        setError('Failed to load strategies')
        console.error(err)
      } finally {
        setLoading(false)
      }
    }

    fetchStrategies()
    // Refresh every 10 seconds
    const interval = setInterval(fetchStrategies, 10000)
    return () => clearInterval(interval)
  }, [])

  const passedCount = strategies.filter(s => s.status === 'passed' || s.status === 'approved').length
  const failedCount = strategies.filter(s => s.status === 'rejected' || s.status === 'failed').length

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.6, delay: 0.6 }}
      className="border border-terminal-border bg-terminal-panel/50 backdrop-blur-sm rounded-lg p-6"
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-1 h-6 bg-terminal-amber" />
          <h2 className="text-xl font-display font-bold text-terminal-bright tracking-wide">
            STRATEGY CANDIDATES
          </h2>
        </div>
        <div className="text-xs font-mono text-terminal-text/60">
          {strategies.length} VALIDATED
        </div>
      </div>

      {/* Loading State */}
      {loading && strategies.length === 0 && (
        <div className="text-center py-8">
          <div className="text-terminal-text/60 font-mono text-sm">Loading strategies...</div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="text-center py-8">
          <div className="text-terminal-red font-mono text-sm">{error}</div>
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && strategies.length === 0 && (
        <div className="text-center py-8">
          <div className="text-terminal-text/60 font-mono text-sm mb-2">No strategies validated yet</div>
          <div className="text-terminal-text/40 font-mono text-xs">Run the Strategy Factory to generate candidates</div>
        </div>
      )}

      {/* Strategy List */}
      {strategies.length > 0 && (
        <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
          {strategies.map((strategy, idx) => {
            const config = statusConfig[strategy.status as keyof typeof statusConfig] || statusConfig.testing

            return (
              <motion.div
                key={strategy.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: idx * 0.05 }}
                onClick={() => setSelectedStrategy(strategy.id)}
                className={`
                  border ${selectedStrategy === strategy.id ? config.border : 'border-terminal-border'}
                  bg-terminal-bg/50 p-4 rounded cursor-pointer
                  hover:bg-terminal-bg/80 transition-all duration-300
                  group
                `}
              >
                {/* Header */}
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-sm font-mono font-semibold text-terminal-bright">
                        {strategy.name}
                      </span>
                      {strategy.status === 'deployed' && (
                        <div className="w-1.5 h-1.5 bg-terminal-green rounded-full animate-pulse" />
                      )}
                    </div>
                    <div className="text-[10px] font-mono text-terminal-text/40">
                      {strategy.type} • ID: {strategy.id.slice(0, 8)}
                    </div>
                  </div>
                  <div className={`px-2 py-1 ${config.bg} border ${config.border} rounded text-[10px] font-mono ${config.color} font-semibold`}>
                    {config.label}
                  </div>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-3 gap-3 mb-3">
                  <div>
                    <div className="text-[9px] font-mono text-terminal-text/40 mb-0.5">SHARPE</div>
                    <div className={`text-sm font-mono font-bold metric-value ${strategy.sharpe > 1.0 ? 'text-terminal-green' : 'text-terminal-amber'}`}>
                      {strategy.sharpe.toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <div className="text-[9px] font-mono text-terminal-text/40 mb-0.5">DSR</div>
                    <div className={`text-sm font-mono font-bold metric-value ${strategy.dsr >= 0.95 ? 'text-terminal-green' : 'text-terminal-red'}`}>
                      {strategy.dsr.toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <div className="text-[9px] font-mono text-terminal-text/40 mb-0.5">RETURN</div>
                    <div className={`text-sm font-mono font-bold metric-value ${(strategy.returns || 0) > 0 ? 'text-terminal-green' : 'text-terminal-red'}`}>
                      {(strategy.returns || 0).toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="w-full h-1 bg-terminal-border rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(100, (strategy.sharpe / 1.5) * 100)}%` }}
                    transition={{ duration: 0.8, delay: idx * 0.05 }}
                    className={`h-full ${config.bg.replace('/10', '')}`}
                  />
                </div>

                {/* Expanded Info */}
                {selectedStrategy === strategy.id && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="mt-3 pt-3 border-t border-terminal-border text-xs font-mono text-terminal-text/60 space-y-1"
                  >
                    <div>Type: {strategy.type}</div>
                    <div>DSR p-value: {strategy.dsr.toFixed(4)}</div>
                    <div>Annual Return: {(strategy.annual_return * 100).toFixed(2)}%</div>
                  </motion.div>
                )}
              </motion.div>
            )
          })}
        </div>
      )}

      {/* Footer Stats */}
      <div className="mt-6 pt-6 border-t border-terminal-border grid grid-cols-3 gap-4 text-center">
        <div>
          <div className="text-[10px] font-mono text-terminal-text/40 mb-1">TOTAL</div>
          <div className="text-lg font-mono font-bold text-terminal-bright metric-value">{strategies.length}</div>
        </div>
        <div>
          <div className="text-[10px] font-mono text-terminal-text/40 mb-1">PASSED</div>
          <div className="text-lg font-mono font-bold text-terminal-green metric-value">{passedCount}</div>
        </div>
        <div>
          <div className="text-[10px] font-mono text-terminal-text/40 mb-1">FAILED</div>
          <div className="text-lg font-mono font-bold text-terminal-red metric-value">{failedCount}</div>
        </div>
      </div>
    </motion.div>
  )
}
