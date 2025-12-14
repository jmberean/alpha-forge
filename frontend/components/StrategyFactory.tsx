'use client'

import { motion } from 'framer-motion'
import { useState } from 'react'
import { runFactory, pollFactoryResult, FactoryResult } from '@/lib/api'

export default function StrategyFactory() {
  const [symbol, setSymbol] = useState('SPY')
  const [populationSize, setPopulationSize] = useState(30)
  const [generations, setGenerations] = useState(5)
  const [validateTopN, setValidateTopN] = useState(5)
  const [isRunning, setIsRunning] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const [result, setResult] = useState<FactoryResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleRun = async () => {
    setIsRunning(true)
    setLogs([])
    setResult(null)
    setError(null)

    try {
      // Start factory
      const response = await runFactory({
        symbol,
        population_size: populationSize,
        generations,
        validate_top_n: validateTopN,
      })

      setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Factory started: ${response.factory_id}`])

      // Poll for results
      const finalResult = await pollFactoryResult(
        response.factory_id,
        (update) => {
          if (update.logs) {
            setLogs(update.logs)
          }
        },
        2000
      )

      setResult(finalResult)
      if (finalResult.status === 'failed') {
        setError('Factory run failed')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setIsRunning(false)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="border border-terminal-cyan bg-terminal-panel/50 backdrop-blur-sm rounded-lg p-6"
    >
      <div className="flex items-center gap-3 mb-6">
        <div className="w-1 h-6 bg-terminal-cyan" />
        <h2 className="text-xl font-display font-bold text-terminal-bright tracking-wide">
          STRATEGY FACTORY
        </h2>
        <span className="text-[10px] font-mono text-terminal-cyan bg-terminal-cyan/10 px-2 py-0.5 rounded">
          GENETIC EVOLUTION
        </span>
      </div>

      {/* Configuration */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
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
            onChange={(e) => setPopulationSize(parseInt(e.target.value) || 30)}
            disabled={isRunning}
            min={10}
            max={200}
            className="w-full bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-sm font-mono text-terminal-bright focus:border-terminal-cyan focus:outline-none disabled:opacity-50"
          />
        </div>
        <div>
          <label className="block text-[10px] font-mono text-terminal-text/60 mb-2">GENERATIONS</label>
          <input
            type="number"
            value={generations}
            onChange={(e) => setGenerations(parseInt(e.target.value) || 5)}
            disabled={isRunning}
            min={1}
            max={100}
            className="w-full bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-sm font-mono text-terminal-bright focus:border-terminal-cyan focus:outline-none disabled:opacity-50"
          />
        </div>
        <div>
          <label className="block text-[10px] font-mono text-terminal-text/60 mb-2">VALIDATE TOP</label>
          <input
            type="number"
            value={validateTopN}
            onChange={(e) => setValidateTopN(parseInt(e.target.value) || 5)}
            disabled={isRunning}
            min={1}
            max={20}
            className="w-full bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-sm font-mono text-terminal-bright focus:border-terminal-cyan focus:outline-none disabled:opacity-50"
          />
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
            RUNNING EVOLUTION...
          </span>
        ) : (
          'START FACTORY'
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

      {/* Results */}
      {result && result.status === 'completed' && (
        <div className="mt-6">
          <div className="text-[10px] font-mono text-terminal-text/60 mb-2">RESULTS</div>
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="bg-terminal-bg border border-terminal-border rounded p-4 text-center">
              <div className="text-2xl font-mono font-bold text-terminal-bright">{result.stats.generated}</div>
              <div className="text-[10px] font-mono text-terminal-text/60">GENERATED</div>
            </div>
            <div className="bg-terminal-bg border border-terminal-border rounded p-4 text-center">
              <div className="text-2xl font-mono font-bold text-terminal-amber">{result.stats.validated}</div>
              <div className="text-[10px] font-mono text-terminal-text/60">VALIDATED</div>
            </div>
            <div className="bg-terminal-bg border border-terminal-border rounded p-4 text-center">
              <div className={`text-2xl font-mono font-bold ${result.stats.passed > 0 ? 'text-terminal-green' : 'text-terminal-red'}`}>
                {result.stats.passed}
              </div>
              <div className="text-[10px] font-mono text-terminal-text/60">PASSED</div>
            </div>
          </div>

          {/* Strategy Summary */}
          {result.strategies.length > 0 && (
            <div className="space-y-2">
              <div className="text-[10px] font-mono text-terminal-text/60 mb-2">TOP STRATEGIES</div>
              {result.strategies.slice(0, 5).map((strategy, idx) => (
                <div
                  key={strategy.id}
                  className={`
                    flex items-center justify-between p-3 rounded border
                    ${strategy.status === 'passed'
                      ? 'border-terminal-green bg-terminal-green/5'
                      : 'border-terminal-red bg-terminal-red/5'
                    }
                  `}
                >
                  <div>
                    <div className="text-sm font-mono font-semibold text-terminal-bright">
                      #{idx + 1} {strategy.name}
                    </div>
                    <div className="text-[10px] font-mono text-terminal-text/60">
                      Sharpe: {strategy.sharpe.toFixed(2)} | DSR: {strategy.dsr.toFixed(3)}
                    </div>
                  </div>
                  <div className={`
                    px-2 py-1 rounded text-[10px] font-mono font-semibold
                    ${strategy.status === 'passed'
                      ? 'bg-terminal-green/20 text-terminal-green'
                      : 'bg-terminal-red/20 text-terminal-red'
                    }
                  `}>
                    {strategy.status === 'passed' ? 'PASSED' : 'FAILED'}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-4 p-4 bg-terminal-red/10 border border-terminal-red rounded">
          <div className="text-terminal-red font-mono text-sm">{error}</div>
        </div>
      )}
    </motion.div>
  )
}
