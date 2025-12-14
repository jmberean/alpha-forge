'use client'

import { motion } from 'framer-motion'

interface Metric {
  label: string
  value: string | number
  threshold: string
  status: 'pass' | 'warning' | 'fail'
  description: string
}

const metrics: Metric[] = [
  {
    label: 'Deflated Sharpe Ratio',
    value: '0.96',
    threshold: '> 0.95',
    status: 'pass',
    description: 'Accounts for multiple testing (1000 trials)'
  },
  {
    label: 'Probability of Backtest Overfitting',
    value: '0.04',
    threshold: '< 0.05',
    status: 'pass',
    description: 'CPCV with 12,870 train/test combinations'
  },
  {
    label: 'Sharpe Ratio',
    value: '1.23',
    threshold: '> 1.0',
    status: 'pass',
    description: 'Risk-adjusted returns (annualized)'
  },
  {
    label: 'Implementation Shortfall',
    value: '22%',
    threshold: '< 30%',
    status: 'pass',
    description: 'Vectorized vs event-driven degradation'
  },
  {
    label: 'SPA p-value',
    value: '0.03',
    threshold: '< 0.05',
    status: 'pass',
    description: "Hansen's Superior Predictive Ability test"
  },
  {
    label: 'Stress Test Pass Rate',
    value: '83%',
    threshold: '≥ 80%',
    status: 'pass',
    description: '5/6 scenarios passed (1 crisis failed)'
  },
  {
    label: 'Annual Return',
    value: '18.4%',
    threshold: '> 10%',
    status: 'pass',
    description: 'Compounded annual growth rate'
  },
  {
    label: 'Max Drawdown',
    value: '-12.3%',
    threshold: '< -25%',
    status: 'pass',
    description: 'Largest peak-to-trough decline'
  },
]

const statusColors = {
  pass: 'border-terminal-green text-terminal-green',
  warning: 'border-terminal-amber text-terminal-amber',
  fail: 'border-terminal-red text-terminal-red',
}

export default function MetricsGrid() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.4 }}
      className="border border-terminal-border bg-terminal-panel/50 backdrop-blur-sm rounded-lg p-8"
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-1 h-6 bg-terminal-amber" />
          <h2 className="text-2xl font-display font-bold text-terminal-bright tracking-wide">
            VALIDATION METRICS
          </h2>
        </div>
        <div className="px-3 py-1 bg-terminal-green/10 border border-terminal-green rounded text-xs font-mono text-terminal-green">
          ✓ ALL TESTS PASSED
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((metric, idx) => (
          <motion.div
            key={metric.label}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: 0.5 + idx * 0.05 }}
            className={`
              border ${statusColors[metric.status]}
              bg-terminal-bg/50 p-4 rounded
              hover:bg-terminal-bg/80 transition-all duration-300
              group cursor-pointer
            `}
          >
            {/* Status Indicator */}
            <div className="flex items-center justify-between mb-3">
              <div className="text-[10px] font-mono text-terminal-text/40">
                {metric.threshold}
              </div>
              <div className={`w-2 h-2 rounded-full ${statusColors[metric.status].replace('border-', 'bg-').replace('text-', 'bg-')} animate-pulse`} />
            </div>

            {/* Metric Label */}
            <div className="text-xs font-mono text-terminal-text/80 mb-2 uppercase tracking-wide">
              {metric.label}
            </div>

            {/* Metric Value */}
            <div className={`text-3xl font-mono font-bold ${statusColors[metric.status]} metric-value mb-3`}>
              {metric.value}
            </div>

            {/* Description (shown on hover) */}
            <div className="text-xs font-mono text-terminal-text/40 opacity-0 group-hover:opacity-100 transition-opacity duration-300 h-8">
              {metric.description}
            </div>

            {/* Visual Bar */}
            <div className="mt-2 w-full h-0.5 bg-terminal-border rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: '100%' }}
                transition={{ duration: 0.8, delay: 0.6 + idx * 0.05 }}
                className={`h-full ${statusColors[metric.status].replace('border-', 'bg-').replace('text-', 'bg-')}`}
              />
            </div>
          </motion.div>
        ))}
      </div>

      {/* Summary Footer */}
      <div className="mt-6 pt-6 border-t border-terminal-border">
        <div className="flex items-center justify-between">
          <div className="text-xs font-mono text-terminal-text/60">
            Last validation: <span className="text-terminal-amber">2024-12-13 18:42:17 UTC</span>
          </div>
          <div className="flex items-center gap-2 text-xs font-mono">
            <span className="text-terminal-green">▲</span>
            <span className="text-terminal-text/60">Recommendation:</span>
            <span className="text-terminal-green font-semibold">DEPLOY WITH APPROVAL</span>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
