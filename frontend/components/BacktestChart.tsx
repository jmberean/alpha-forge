'use client'

import { motion } from 'framer-motion'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

interface ChartDataPoint {
  date: string
  equity: number
  benchmark: number
}

interface ChartMetrics {
  total_return: number
  num_trades?: number
  win_rate?: number
  profit_factor?: number
  avg_win?: number
  avg_loss?: number
}

interface BacktestChartProps {
  data?: ChartDataPoint[]
  metrics?: ChartMetrics
}

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-terminal-panel border border-terminal-border p-3 rounded font-mono text-xs">
        <div className="text-terminal-text/60 mb-1">{payload[0].payload.date}</div>
        <div className="text-terminal-green">
          Strategy: <span className="metric-value font-semibold">${payload[0].value.toLocaleString()}</span>
        </div>
        <div className="text-terminal-amber">
          Benchmark: <span className="metric-value">${payload[1].value.toLocaleString()}</span>
        </div>
      </div>
    )
  }
  return null
}

export default function BacktestChart({ data = [], metrics }: BacktestChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="border border-terminal-border bg-terminal-panel/50 backdrop-blur-sm rounded-lg p-6 flex items-center justify-center h-64">
        <span className="text-terminal-text/60 font-mono text-sm">No data available</span>
      </div>
    )
  }

  const finalEquity = data[data.length - 1].equity
  const initialEquity = data[0].equity
  
  // Use passed metric if available, otherwise calculate
  const totalReturn = metrics?.total_return 
    ? (metrics.total_return * 100).toFixed(1)
    : ((finalEquity - initialEquity) / initialEquity * 100).toFixed(1)

  // Calculate period in years
  const start = new Date(data[0].date)
  const end = new Date(data[data.length - 1].date)
  const years = (end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24 * 365.25)

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.6, delay: 0.6 }}
      className="border border-terminal-border bg-terminal-panel/50 backdrop-blur-sm rounded-lg p-6"
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-1 h-6 bg-terminal-green" />
          <h2 className="text-xl font-display font-bold text-terminal-bright tracking-wide">
            EQUITY CURVE
          </h2>
        </div>
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-terminal-green" />
            <span className="text-terminal-text/60">Strategy</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-terminal-amber" />
            <span className="text-terminal-text/60">Benchmark</span>
          </div>
        </div>
      </div>

      {/* Chart Statistics */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-terminal-bg/50 border border-terminal-border rounded p-3">
          <div className="text-[10px] font-mono text-terminal-text/40 mb-1">FINAL VALUE</div>
          <div className="text-lg font-mono font-bold text-terminal-green metric-value">
            ${finalEquity.toLocaleString()}
          </div>
        </div>
        <div className="bg-terminal-bg/50 border border-terminal-border rounded p-3">
          <div className="text-[10px] font-mono text-terminal-text/40 mb-1">TOTAL RETURN</div>
          <div className="text-lg font-mono font-bold text-terminal-green metric-value">
            {Number(totalReturn) > 0 ? '+' : ''}{totalReturn}%
          </div>
        </div>
        <div className="bg-terminal-bg/50 border border-terminal-border rounded p-3">
          <div className="text-[10px] font-mono text-terminal-text/40 mb-1">PERIOD</div>
          <div className="text-lg font-mono font-bold text-terminal-amber metric-value">
            {years.toFixed(1)}Y
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="h-64 -mx-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <XAxis
              dataKey="date"
              stroke="#1a1f29"
              tick={{ fill: '#b3b8c4', fontSize: 10, fontFamily: 'JetBrains Mono' }}
              tickFormatter={(value) => new Date(value).getFullYear().toString()}
              minTickGap={30}
            />
            <YAxis
              stroke="#1a1f29"
              tick={{ fill: '#b3b8c4', fontSize: 10, fontFamily: 'JetBrains Mono' }}
              tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
              domain={['auto', 'auto']}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={initialEquity} stroke="#1a1f29" strokeDasharray="3 3" />
            <Line
              type="monotone"
              dataKey="benchmark"
              stroke="#ffb454"
              strokeWidth={1.5}
              dot={false}
              opacity={0.6}
            />
            <Line
              type="monotone"
              dataKey="equity"
              stroke="#7fd962"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Trade Stats (only if metrics provided) */}
      {metrics && metrics.num_trades !== undefined && (
        <div className="mt-6 pt-6 border-t border-terminal-border grid grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-[10px] font-mono text-terminal-text/40 mb-1">TRADES</div>
            <div className="text-sm font-mono font-semibold text-terminal-text metric-value">
              {metrics.num_trades}
            </div>
          </div>
          <div>
            <div className="text-[10px] font-mono text-terminal-text/40 mb-1">WIN RATE</div>
            <div className="text-sm font-mono font-semibold text-terminal-green metric-value">
              {(metrics.win_rate! * 100).toFixed(1)}%
            </div>
          </div>
          <div>
            <div className="text-[10px] font-mono text-terminal-text/40 mb-1">PROFIT FACTOR</div>
            <div className="text-sm font-mono font-semibold text-terminal-green metric-value">
              {metrics.profit_factor?.toFixed(2) || '-'}
            </div>
          </div>
          <div>
            <div className="text-[10px] font-mono text-terminal-text/40 mb-1">AVG WIN/LOSS</div>
            <div className="text-sm font-mono font-semibold text-terminal-amber metric-value">
              {(metrics.avg_win && metrics.avg_loss) 
                ? (metrics.avg_win / metrics.avg_loss).toFixed(2) 
                : '-'}
            </div>
          </div>
        </div>
      )}
    </motion.div>
  )
}