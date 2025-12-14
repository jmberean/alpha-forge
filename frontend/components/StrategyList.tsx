'use client'

import { motion } from 'framer-motion'
import { useState } from 'react'

interface Strategy {
  id: string
  name: string
  type: string
  sharpe: number
  dsr: number
  pbo: number
  status: 'deployed' | 'approved' | 'testing' | 'rejected'
  returns: number
}

const strategies: Strategy[] = [
  {
    id: 'strat_001',
    name: 'SMA_Cross_Genetic_v47',
    type: 'Momentum',
    sharpe: 1.23,
    dsr: 0.96,
    pbo: 0.04,
    status: 'deployed',
    returns: 18.4,
  },
  {
    id: 'strat_002',
    name: 'MeanRev_RSI_Optimized',
    type: 'Mean Reversion',
    sharpe: 1.18,
    dsr: 0.95,
    pbo: 0.048,
    status: 'approved',
    returns: 16.2,
  },
  {
    id: 'strat_003',
    name: 'Breakout_Volatility_ML',
    type: 'Breakout',
    sharpe: 1.31,
    dsr: 0.97,
    pbo: 0.03,
    status: 'deployed',
    returns: 21.7,
  },
  {
    id: 'strat_004',
    name: 'Pairs_Statistical_Arb',
    type: 'Statistical Arb',
    sharpe: 1.09,
    dsr: 0.93,
    pbo: 0.07,
    status: 'testing',
    returns: 14.8,
  },
  {
    id: 'strat_005',
    name: 'Trend_Follow_Ichimoku',
    type: 'Trend Following',
    sharpe: 1.42,
    dsr: 0.98,
    pbo: 0.02,
    status: 'deployed',
    returns: 24.3,
  },
  {
    id: 'strat_006',
    name: 'Vol_Targeting_Dynamic',
    type: 'Volatility',
    sharpe: 0.87,
    dsr: 0.89,
    pbo: 0.12,
    status: 'rejected',
    returns: 9.2,
  },
]

const statusConfig = {
  deployed: { color: 'text-terminal-green', bg: 'bg-terminal-green/10', border: 'border-terminal-green', label: 'LIVE' },
  approved: { color: 'text-terminal-amber', bg: 'bg-terminal-amber/10', border: 'border-terminal-amber', label: 'APPROVED' },
  testing: { color: 'text-terminal-blue', bg: 'bg-terminal-blue/10', border: 'border-terminal-blue', label: 'TESTING' },
  rejected: { color: 'text-terminal-red', bg: 'bg-terminal-red/10', border: 'border-terminal-red', label: 'FAILED' },
}

export default function StrategyList() {
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null)

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
          TOP 6 / 10,000
        </div>
      </div>

      {/* Strategy List */}
      <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
        {strategies.map((strategy, idx) => {
          const config = statusConfig[strategy.status]

          return (
            <motion.div
              key={strategy.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: 0.7 + idx * 0.05 }}
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
                    {strategy.type} • ID: {strategy.id}
                  </div>
                </div>
                <div className={`px-2 py-1 ${config.bg} border ${config.border} rounded text-[10px] font-mono ${config.color} font-semibold`}>
                  {config.label}
                </div>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-4 gap-3 mb-3">
                <div>
                  <div className="text-[9px] font-mono text-terminal-text/40 mb-0.5">SHARPE</div>
                  <div className={`text-sm font-mono font-bold metric-value ${strategy.sharpe > 1.2 ? 'text-terminal-green' : 'text-terminal-amber'}`}>
                    {strategy.sharpe.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div className="text-[9px] font-mono text-terminal-text/40 mb-0.5">DSR</div>
                  <div className={`text-sm font-mono font-bold metric-value ${strategy.dsr >= 0.95 ? 'text-terminal-green' : 'text-terminal-red'}`}>
                    {strategy.dsr.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div className="text-[9px] font-mono text-terminal-text/40 mb-0.5">PBO</div>
                  <div className={`text-sm font-mono font-bold metric-value ${strategy.pbo < 0.05 ? 'text-terminal-green' : 'text-terminal-red'}`}>
                    {strategy.pbo.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div className="text-[9px] font-mono text-terminal-text/40 mb-0.5">RETURN</div>
                  <div className="text-sm font-mono font-bold text-terminal-green metric-value">
                    {strategy.returns.toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Progress Bar */}
              <div className="w-full h-1 bg-terminal-border rounded-full overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(strategy.sharpe / 1.5) * 100}%` }}
                  transition={{ duration: 0.8, delay: 0.8 + idx * 0.05 }}
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
                  <div>• Generated via genetic programming (generation 47)</div>
                  <div>• Tested on SPY 2020-2024 (4 years)</div>
                  <div>• CPCV validated with 12,870 combinations</div>
                  <div>• Event-driven backtest: 22% implementation shortfall</div>
                </motion.div>
              )}
            </motion.div>
          )
        })}
      </div>

      {/* Footer Stats */}
      <div className="mt-6 pt-6 border-t border-terminal-border grid grid-cols-3 gap-4 text-center">
        <div>
          <div className="text-[10px] font-mono text-terminal-text/40 mb-1">DEPLOYED</div>
          <div className="text-lg font-mono font-bold text-terminal-green metric-value">3</div>
        </div>
        <div>
          <div className="text-[10px] font-mono text-terminal-text/40 mb-1">IN REVIEW</div>
          <div className="text-lg font-mono font-bold text-terminal-amber metric-value">1</div>
        </div>
        <div>
          <div className="text-[10px] font-mono text-terminal-text/40 mb-1">TESTING</div>
          <div className="text-lg font-mono font-bold text-terminal-blue metric-value">1</div>
        </div>
      </div>
    </motion.div>
  )
}
