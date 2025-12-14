'use client'

import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'
import { getLatestMetrics, MetricsData } from '@/lib/api'

interface DisplayMetric {
  label: string
  value: string | number
  threshold: string
  status: 'pass' | 'warning' | 'fail'
  description: string
}

const statusColors = {
  pass: 'border-terminal-green text-terminal-green',
  warning: 'border-terminal-amber text-terminal-amber',
  fail: 'border-terminal-red text-terminal-red',
}

function determineStatus(name: string, value: number): 'pass' | 'warning' | 'fail' {
  if (name.includes('Deflated Sharpe') || name.includes('DSR')) {
    return value > 0.95 ? 'pass' : value > 0.9 ? 'warning' : 'fail'
  }
  if (name.includes('PBO') || name.includes('Overfitting')) {
    return value < 0.05 ? 'pass' : value < 0.1 ? 'warning' : 'fail'
  }
  if (name.includes('Sharpe Ratio') && !name.includes('Deflated')) {
    return value > 1.0 ? 'pass' : value > 0.5 ? 'warning' : 'fail'
  }
  if (name.includes('Annual Return')) {
    return value > 10 ? 'pass' : value > 0 ? 'warning' : 'fail'
  }
  if (name.includes('Max Drawdown')) {
    return value < 25 ? 'pass' : value < 40 ? 'warning' : 'fail'
  }
  return 'warning'
}

function formatValue(value: number, unit: string): string {
  if (unit === '%') {
    return `${value.toFixed(1)}%`
  }
  if (value < 1 && value > 0) {
    return value.toFixed(3)
  }
  return value.toFixed(2)
}

export default function MetricsGrid() {
  const [metricsData, setMetricsData] = useState<MetricsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const data = await getLatestMetrics()
        setMetricsData(data)
        setError(null)
      } catch (err) {
        setError('Failed to load metrics')
        console.error(err)
      } finally {
        setLoading(false)
      }
    }

    fetchMetrics()
    // Refresh every 10 seconds
    const interval = setInterval(fetchMetrics, 10000)
    return () => clearInterval(interval)
  }, [])

  const displayMetrics: DisplayMetric[] = metricsData?.metrics.map(m => ({
    label: m.name,
    value: formatValue(m.value, m.unit),
    threshold: m.threshold || '',
    status: determineStatus(m.name, m.value),
    description: m.threshold ? `Threshold: ${m.threshold}` : '',
  })) || []

  const allPassed = displayMetrics.every(m => m.status === 'pass')
  const anyFailed = displayMetrics.some(m => m.status === 'fail')

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
        {metricsData?.has_data && (
          <div className={`px-3 py-1 ${allPassed ? 'bg-terminal-green/10 border-terminal-green text-terminal-green' : anyFailed ? 'bg-terminal-red/10 border-terminal-red text-terminal-red' : 'bg-terminal-amber/10 border-terminal-amber text-terminal-amber'} border rounded text-xs font-mono`}>
            {allPassed ? '✓ ALL TESTS PASSED' : anyFailed ? '✗ VALIDATION FAILED' : '⚠ NEEDS REVIEW'}
          </div>
        )}
      </div>

      {/* Loading State */}
      {loading && (
        <div className="text-center py-8">
          <div className="text-terminal-text/60 font-mono text-sm">Loading metrics...</div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="text-center py-8">
          <div className="text-terminal-red font-mono text-sm">{error}</div>
        </div>
      )}

      {/* No Data State */}
      {!loading && !error && !metricsData?.has_data && (
        <div className="text-center py-8">
          <div className="text-terminal-text/60 font-mono text-sm mb-2">No validation metrics yet</div>
          <div className="text-terminal-text/40 font-mono text-xs">Run a validation or the Strategy Factory to see metrics</div>
        </div>
      )}

      {/* Metrics Grid */}
      {metricsData?.has_data && displayMetrics.length > 0 && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {displayMetrics.map((metric, idx) => (
              <motion.div
                key={metric.label}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3, delay: idx * 0.05 }}
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
                    transition={{ duration: 0.8, delay: idx * 0.05 }}
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
                Strategy: <span className="text-terminal-amber">{metricsData.strategy_name}</span>
                {metricsData.timestamp && (
                  <span className="ml-4">Last updated: {new Date(metricsData.timestamp).toLocaleString()}</span>
                )}
              </div>
              <div className="flex items-center gap-2 text-xs font-mono">
                <span className={allPassed ? 'text-terminal-green' : anyFailed ? 'text-terminal-red' : 'text-terminal-amber'}>
                  {allPassed ? '▲' : anyFailed ? '▼' : '►'}
                </span>
                <span className="text-terminal-text/60">Recommendation:</span>
                <span className={`font-semibold ${allPassed ? 'text-terminal-green' : anyFailed ? 'text-terminal-red' : 'text-terminal-amber'}`}>
                  {allPassed ? 'DEPLOY WITH APPROVAL' : anyFailed ? 'DO NOT DEPLOY' : 'NEEDS REVIEW'}
                </span>
              </div>
            </div>
          </div>
        </>
      )}
    </motion.div>
  )
}
