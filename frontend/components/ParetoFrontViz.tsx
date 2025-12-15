'use client'

import { motion } from 'framer-motion'
import { useState } from 'react'
import { ExpressionStrategy } from '@/lib/api'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

interface ParetoFrontVizProps {
  strategies: ExpressionStrategy[]
}

export default function ParetoFrontViz({ strategies }: ParetoFrontVizProps) {
  const [xAxis, setXAxis] = useState<'sharpe' | 'drawdown' | 'turnover' | 'complexity'>('sharpe')
  const [yAxis, setYAxis] = useState<'sharpe' | 'drawdown' | 'turnover' | 'complexity'>('drawdown')
  const [selectedStrategy, setSelectedStrategy] = useState<ExpressionStrategy | null>(null)

  // Prepare data for chart
  const chartData = strategies.map((strategy, idx) => ({
    x: strategy.fitness[xAxis],
    y: strategy.fitness[yAxis],
    size: strategy.size,
    complexity: strategy.complexity,
    formula: strategy.formula,
    strategyIndex: idx,
  }))

  const handleAxisChange = (axis: 'x' | 'y', value: string) => {
    if (axis === 'x') {
      setXAxis(value as typeof xAxis)
    } else {
      setYAxis(value as typeof yAxis)
    }
  }

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="bg-terminal-bg border border-terminal-cyan rounded p-3 shadow-lg">
          <div className="text-xs font-mono text-terminal-bright mb-2">
            Strategy #{data.strategyIndex + 1}
          </div>
          <div className="text-[10px] font-mono space-y-1">
            <div className="text-terminal-cyan">
              {xAxis}: {data.x.toFixed(3)}
            </div>
            <div className="text-terminal-amber">
              {yAxis}: {data.y.toFixed(3)}
            </div>
            <div className="text-terminal-text/60">
              Size: {data.size} | Complexity: {data.complexity.toFixed(3)}
            </div>
          </div>
        </div>
      )
    }
    return null
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.3 }}
      className="border border-terminal-cyan bg-terminal-panel/50 backdrop-blur-sm rounded-lg p-6"
    >
      <div className="flex items-center gap-3 mb-6">
        <div className="w-1 h-6 bg-terminal-cyan" />
        <h3 className="text-lg font-display font-bold text-terminal-bright">PARETO FRONT</h3>
        <span className="text-[10px] font-mono text-terminal-text/60">
          {strategies.length} Non-Dominated Strategies
        </span>
      </div>

      {/* Axis Selection */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block text-[10px] font-mono text-terminal-text/60 mb-2">X-AXIS</label>
          <select
            value={xAxis}
            onChange={(e) => handleAxisChange('x', e.target.value)}
            className="w-full bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-sm font-mono text-terminal-bright focus:border-terminal-cyan focus:outline-none"
          >
            <option value="sharpe">Sharpe Ratio</option>
            <option value="drawdown">Max Drawdown</option>
            <option value="turnover">Turnover</option>
            <option value="complexity">Complexity</option>
          </select>
        </div>
        <div>
          <label className="block text-[10px] font-mono text-terminal-text/60 mb-2">Y-AXIS</label>
          <select
            value={yAxis}
            onChange={(e) => handleAxisChange('y', e.target.value)}
            className="w-full bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-sm font-mono text-terminal-bright focus:border-terminal-cyan focus:outline-none"
          >
            <option value="sharpe">Sharpe Ratio</option>
            <option value="drawdown">Max Drawdown</option>
            <option value="turnover">Turnover</option>
            <option value="complexity">Complexity</option>
          </select>
        </div>
      </div>

      {/* Scatter Plot */}
      <div className="bg-terminal-bg border border-terminal-border rounded p-4 mb-6">
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a2332" />
            <XAxis
              type="number"
              dataKey="x"
              name={xAxis}
              stroke="#64ffda"
              tick={{ fill: '#64ffda', fontSize: 11, fontFamily: 'monospace' }}
              label={{ value: xAxis.toUpperCase(), position: 'insideBottom', offset: -10, fill: '#64ffda', fontSize: 12 }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name={yAxis}
              stroke="#64ffda"
              tick={{ fill: '#64ffda', fontSize: 11, fontFamily: 'monospace' }}
              label={{ value: yAxis.toUpperCase(), angle: -90, position: 'insideLeft', fill: '#64ffda', fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
            <Scatter
              name="Strategies"
              data={chartData}
              fill="#64ffda"
              fillOpacity={0.6}
              onClick={(data) => {
                if (data && data.strategyIndex !== undefined) {
                  setSelectedStrategy(strategies[data.strategyIndex])
                }
              }}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Selected Strategy Details */}
      {selectedStrategy && (
        <div className="bg-terminal-bg border border-terminal-cyan rounded p-4">
          <div className="flex items-start justify-between mb-3">
            <div className="text-sm font-mono font-semibold text-terminal-cyan">SELECTED STRATEGY</div>
            <button
              onClick={() => setSelectedStrategy(null)}
              className="text-xs font-mono text-terminal-text/60 hover:text-terminal-bright"
            >
              [CLOSE]
            </button>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
            <div>
              <div className="text-[10px] font-mono text-terminal-text/60">SHARPE</div>
              <div className="text-sm font-mono font-bold text-terminal-cyan">
                {selectedStrategy.fitness.sharpe.toFixed(3)}
              </div>
            </div>
            <div>
              <div className="text-[10px] font-mono text-terminal-text/60">DRAWDOWN</div>
              <div className="text-sm font-mono font-bold text-terminal-amber">
                {selectedStrategy.fitness.drawdown.toFixed(3)}
              </div>
            </div>
            <div>
              <div className="text-[10px] font-mono text-terminal-text/60">TURNOVER</div>
              <div className="text-sm font-mono font-bold text-terminal-green">
                {selectedStrategy.fitness.turnover.toFixed(3)}
              </div>
            </div>
            <div>
              <div className="text-[10px] font-mono text-terminal-text/60">COMPLEXITY</div>
              <div className="text-sm font-mono font-bold text-terminal-purple">
                {selectedStrategy.fitness.complexity.toFixed(3)}
              </div>
            </div>
          </div>

          <div className="text-[10px] font-mono text-terminal-text/60 mb-1">FORMULA</div>
          <div className="text-xs font-mono text-terminal-text bg-terminal-bg/50 p-3 rounded overflow-x-auto">
            {selectedStrategy.formula}
          </div>

          <div className="grid grid-cols-3 gap-3 mt-3">
            <div>
              <div className="text-[10px] font-mono text-terminal-text/60">SIZE</div>
              <div className="text-sm font-mono text-terminal-bright">{selectedStrategy.size} nodes</div>
            </div>
            <div>
              <div className="text-[10px] font-mono text-terminal-text/60">DEPTH</div>
              <div className="text-sm font-mono text-terminal-bright">{selectedStrategy.depth} levels</div>
            </div>
            <div>
              <div className="text-[10px] font-mono text-terminal-text/60">COMPLEXITY</div>
              <div className="text-sm font-mono text-terminal-bright">{selectedStrategy.complexity.toFixed(3)}</div>
            </div>
          </div>
        </div>
      )}

      {/* Info */}
      <div className="mt-6 p-3 bg-terminal-cyan/5 border border-terminal-cyan/20 rounded">
        <div className="text-[10px] font-mono text-terminal-text/60 mb-1">ABOUT PARETO FRONT</div>
        <div className="text-xs font-mono text-terminal-text">
          Non-dominated strategies represent optimal trade-offs between objectives. No strategy is strictly better in all objectives simultaneously.
        </div>
      </div>
    </motion.div>
  )
}
