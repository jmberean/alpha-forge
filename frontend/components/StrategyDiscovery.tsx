'use client'

import { motion } from 'framer-motion'
import { useState } from 'react'
import { runDiscovery, pollDiscoveryResult, DiscoveryResult } from '@/lib/api'
import ParetoFrontViz from './ParetoFrontViz'
import FactorZoo from './FactorZoo'

export default function StrategyDiscovery() {
  const [symbol, setSymbol] = useState('SPY')
  const [populationSize, setPopulationSize] = useState(100)
  const [generations, setGenerations] = useState(20)
  const [minSharpe, setMinSharpe] = useState(0.5)
  const [maxTurnover, setMaxTurnover] = useState(0.2)
  const [maxComplexity, setMaxComplexity] = useState(0.7)
  const [isRunning, setIsRunning] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const [result, setResult] = useState<DiscoveryResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleRun = async () => {
    setIsRunning(true)
    setLogs([])
    setResult(null)
    setError(null)

    try {
      // Start discovery
      const response = await runDiscovery({
        symbol,
        population_size: populationSize,
        n_generations: generations,
        n_objectives: 4,
        min_sharpe: minSharpe,
        max_turnover: maxTurnover,
        max_complexity: maxComplexity,
        validation_split: 0.3,
      })

      setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Discovery started: ${response.discovery_id}`])

      // Poll for results
      const finalResult = await pollDiscoveryResult(
        response.discovery_id,
        (update) => {
          if (update.logs) {
            setLogs(update.logs)
          }
        },
        2000
      )

      setResult(finalResult)
      if (finalResult.status === 'failed') {
        setError(finalResult.error || 'Discovery run failed')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Configuration Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="border border-terminal-cyan bg-terminal-panel/50 backdrop-blur-sm rounded-lg p-6"
      >
        <div className="flex items-center gap-3 mb-6">
          <div className="w-1 h-6 bg-terminal-cyan" />
          <h2 className="text-xl font-display font-bold text-terminal-bright tracking-wide">
            STRATEGY DISCOVERY
          </h2>
          <span className="text-[10px] font-mono text-terminal-cyan bg-terminal-cyan/10 px-2 py-0.5 rounded">
            NSGA-III MULTI-OBJECTIVE
          </span>
        </div>

        {/* Configuration Grid */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
          <div>
            <label className="block text-[10px] font-mono text-terminal-text/60 mb-2">SYMBOL</label>
            <input
              type="text"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              disabled={isRunning}
              className="w-full bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-sm font-mono text-terminal-bright focus:border-terminal-cyan focus:outline-none disabled:opacity-50"
            />
          </div>
          <div>
            <label className="block text-[10px] font-mono text-terminal-text/60 mb-2">POPULATION</label>
            <input
              type="number"
              value={populationSize}
              onChange={(e) => setPopulationSize(parseInt(e.target.value) || 100)}
              disabled={isRunning}
              min={50}
              max={500}
              className="w-full bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-sm font-mono text-terminal-bright focus:border-terminal-cyan focus:outline-none disabled:opacity-50"
            />
          </div>
          <div>
            <label className="block text-[10px] font-mono text-terminal-text/60 mb-2">GENERATIONS</label>
            <input
              type="number"
              value={generations}
              onChange={(e) => setGenerations(parseInt(e.target.value) || 20)}
              disabled={isRunning}
              min={10}
              max={200}
              className="w-full bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-sm font-mono text-terminal-bright focus:border-terminal-cyan focus:outline-none disabled:opacity-50"
            />
          </div>
          <div>
            <label className="block text-[10px] font-mono text-terminal-text/60 mb-2">MIN SHARPE</label>
            <input
              type="number"
              value={minSharpe}
              onChange={(e) => setMinSharpe(parseFloat(e.target.value) || 0.5)}
              disabled={isRunning}
              step={0.1}
              min={0}
              max={3}
              className="w-full bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-sm font-mono text-terminal-bright focus:border-terminal-cyan focus:outline-none disabled:opacity-50"
            />
          </div>
          <div>
            <label className="block text-[10px] font-mono text-terminal-text/60 mb-2">MAX TURNOVER</label>
            <input
              type="number"
              value={maxTurnover}
              onChange={(e) => setMaxTurnover(parseFloat(e.target.value) || 0.2)}
              disabled={isRunning}
              step={0.05}
              min={0.05}
              max={1}
              className="w-full bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-sm font-mono text-terminal-bright focus:border-terminal-cyan focus:outline-none disabled:opacity-50"
            />
          </div>
          <div>
            <label className="block text-[10px] font-mono text-terminal-text/60 mb-2">MAX COMPLEXITY</label>
            <input
              type="number"
              value={maxComplexity}
              onChange={(e) => setMaxComplexity(parseFloat(e.target.value) || 0.7)}
              disabled={isRunning}
              step={0.1}
              min={0.1}
              max={1}
              className="w-full bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-sm font-mono text-terminal-bright focus:border-terminal-cyan focus:outline-none disabled:opacity-50"
            />
          </div>
        </div>

        {/* Objectives Info */}
        <div className="mb-6 p-4 bg-terminal-bg border border-terminal-border rounded">
          <div className="text-[10px] font-mono text-terminal-text/60 mb-2">OPTIMIZATION OBJECTIVES</div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="text-xs font-mono">
              <span className="text-terminal-cyan">①</span> <span className="text-terminal-text">Sharpe Ratio (max)</span>
            </div>
            <div className="text-xs font-mono">
              <span className="text-terminal-amber">②</span> <span className="text-terminal-text">Max Drawdown (min)</span>
            </div>
            <div className="text-xs font-mono">
              <span className="text-terminal-green">③</span> <span className="text-terminal-text">Turnover (min)</span>
            </div>
            <div className="text-xs font-mono">
              <span className="text-terminal-purple">④</span> <span className="text-terminal-text">Complexity (min)</span>
            </div>
          </div>
        </div>

        {/* Run Button */}
        <button
          onClick={handleRun}
          disabled={isRunning}
          className={`
            w-full py-3 rounded font-mono font-semibold text-sm tracking-wider
            transition-all duration-300
            ${isRunning
              ? 'bg-terminal-cyan/20 text-terminal-cyan cursor-not-allowed'
              : 'bg-terminal-cyan text-terminal-bg hover:bg-terminal-cyan/80'
            }
          `}
        >
          {isRunning ? (
            <span className="flex items-center justify-center gap-2">
              <span className="w-4 h-4 border-2 border-terminal-cyan border-t-transparent rounded-full animate-spin" />
              RUNNING NSGA-III...
            </span>
          ) : (
            'START DISCOVERY'
          )}
        </button>

        {/* Logs */}
        {logs.length > 0 && (
          <div className="mt-6">
            <div className="text-[10px] font-mono text-terminal-text/60 mb-2">EXECUTION LOG</div>
            <div className="bg-terminal-bg border border-terminal-border rounded p-4 max-h-48 overflow-y-auto font-mono text-xs">
              {logs.map((log, idx) => (
                <div key={idx} className="text-terminal-text/80 mb-1">
                  {log}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mt-4 p-4 bg-terminal-red/10 border border-terminal-red rounded">
            <div className="text-terminal-red font-mono text-sm">{error}</div>
          </div>
        )}
      </motion.div>

      {/* Results */}
      {result && result.status === 'completed' && (
        <>
          {/* Stats Overview */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-4"
          >
            <div className="border border-terminal-cyan bg-terminal-panel/50 backdrop-blur-sm rounded-lg p-4 text-center">
              <div className="text-2xl font-mono font-bold text-terminal-cyan">{result.pareto_front.length}</div>
              <div className="text-[10px] font-mono text-terminal-text/60">PARETO FRONT</div>
            </div>
            <div className="border border-terminal-amber bg-terminal-panel/50 backdrop-blur-sm rounded-lg p-4 text-center">
              <div className="text-2xl font-mono font-bold text-terminal-amber">{result.factor_zoo.length}</div>
              <div className="text-[10px] font-mono text-terminal-text/60">FACTOR ZOO</div>
            </div>
            <div className="border border-terminal-green bg-terminal-panel/50 backdrop-blur-sm rounded-lg p-4 text-center">
              <div className="text-2xl font-mono font-bold text-terminal-green">
                {Object.keys(result.ensemble_weights).length}
              </div>
              <div className="text-[10px] font-mono text-terminal-text/60">ENSEMBLE</div>
            </div>
            <div className="border border-terminal-purple bg-terminal-panel/50 backdrop-blur-sm rounded-lg p-4 text-center">
              <div className="text-2xl font-mono font-bold text-terminal-purple">{result.generation_stats.length}</div>
              <div className="text-[10px] font-mono text-terminal-text/60">GENERATIONS</div>
            </div>
          </motion.div>

          {/* Best by Objective */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="border border-terminal-cyan bg-terminal-panel/50 backdrop-blur-sm rounded-lg p-6"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className="w-1 h-6 bg-terminal-cyan" />
              <h3 className="text-lg font-display font-bold text-terminal-bright">BEST BY OBJECTIVE</h3>
            </div>

            <div className="space-y-3">
              {Object.entries(result.best_by_objective).map(([objective, strategy]) => (
                <div key={objective} className="bg-terminal-bg border border-terminal-border rounded p-4">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <span className="text-sm font-mono font-semibold text-terminal-cyan uppercase">{objective}</span>
                      <div className="text-xs font-mono text-terminal-text/60 mt-1">
                        Size: {strategy.size} | Depth: {strategy.depth} | Complexity: {strategy.complexity.toFixed(3)}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-mono font-bold text-terminal-bright">
                        {strategy.fitness[objective as keyof typeof strategy.fitness].toFixed(3)}
                      </div>
                    </div>
                  </div>
                  <div className="text-xs font-mono text-terminal-text bg-terminal-bg/50 p-2 rounded overflow-x-auto">
                    {strategy.formula}
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Pareto Front Visualization */}
          <ParetoFrontViz strategies={result.pareto_front} />

          {/* Factor Zoo */}
          <FactorZoo formulas={result.factor_zoo} />
        </>
      )}
    </div>
  )
}
