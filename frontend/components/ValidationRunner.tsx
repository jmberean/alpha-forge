'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { useState } from 'react'
import { validateStrategy, pollValidationResult, type ValidationResult } from '@/lib/api'

export default function ValidationRunner() {
  const [symbol, setSymbol] = useState('SPY')
  const [template, setTemplate] = useState('sma_crossover')
  const [isRunning, setIsRunning] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const [result, setResult] = useState<ValidationResult | null>(null)

  const handleRunValidation = async () => {
    setIsRunning(true)
    setLogs([])
    setResult(null)

    try {
      // Start validation
      const response = await validateStrategy({
        symbol,
        template,
        start_date: '2020-01-01',
        end_date: '2023-12-31',
        n_trials: 100,
        run_cpcv: false,
        run_spa: false,
        run_stress: false,
      })

      setLogs(prev => [...prev, `✓ Validation started: ${response.validation_id}`])

      // Poll for results
      const finalResult = await pollValidationResult(
        response.validation_id,
        (update) => {
          // Update logs as they come in
          if (update.logs && update.logs.length > 0) {
            setLogs(update.logs)
          }
        }
      )

      setResult(finalResult)
      setIsRunning(false)

    } catch (error) {
      setLogs(prev => [...prev, `✗ Error: ${error}`])
      setIsRunning(false)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.8 }}
      className="border border-terminal-border bg-terminal-panel/50 backdrop-blur-sm rounded-lg p-6"
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="w-1 h-6 bg-terminal-cyan" />
          <h2 className="text-xl font-display font-bold text-terminal-bright tracking-wide">
            RUN VALIDATION
          </h2>
        </div>
        {isRunning && (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-terminal-green rounded-full animate-pulse" />
            <span className="text-xs font-mono text-terminal-green">RUNNING</span>
          </div>
        )}
      </div>

      {/* Input Form */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <label className="text-xs font-mono text-terminal-text/60 mb-2 block">SYMBOL</label>
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            disabled={isRunning}
            className="w-full px-3 py-2 bg-terminal-bg border border-terminal-border rounded text-sm font-mono text-terminal-text focus:outline-none focus:border-terminal-green disabled:opacity-50"
          />
        </div>
        <div>
          <label className="text-xs font-mono text-terminal-text/60 mb-2 block">TEMPLATE</label>
          <select
            value={template}
            onChange={(e) => setTemplate(e.target.value)}
            disabled={isRunning}
            className="w-full px-3 py-2 bg-terminal-bg border border-terminal-border rounded text-sm font-mono text-terminal-text focus:outline-none focus:border-terminal-green disabled:opacity-50"
          >
            <option value="sma_crossover">SMA Crossover</option>
            <option value="rsi_mean_reversion">RSI Mean Reversion</option>
            <option value="macd_trend">MACD Trend</option>
            <option value="bollinger_breakout">Bollinger Breakout</option>
            <option value="buy_and_hold">Buy and Hold</option>
          </select>
        </div>
      </div>

      {/* Run Button */}
      <button
        onClick={handleRunValidation}
        disabled={isRunning}
        className="w-full px-4 py-3 bg-terminal-green text-terminal-bg rounded font-mono font-semibold hover:bg-terminal-green/90 transition-all disabled:opacity-50 disabled:cursor-not-allowed mb-6"
      >
        {isRunning ? 'VALIDATING...' : 'START VALIDATION'}
      </button>

      {/* Logs Output */}
      {logs.length > 0 && (
        <div className="bg-terminal-bg border border-terminal-border rounded-lg p-4 max-h-64 overflow-y-auto font-mono text-xs">
          <AnimatePresence>
            {logs.map((log, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className={`${
                  log.includes('✓') ? 'text-terminal-green' :
                  log.includes('✗') || log.includes('FAIL') ? 'text-terminal-red' :
                  log.includes('⚠') ? 'text-terminal-amber' :
                  'text-terminal-text'
                }`}
              >
                {log}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}

      {/* Results Summary */}
      {result && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="mt-6 p-4 border border-terminal-border rounded-lg bg-terminal-bg/50"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-mono font-bold text-terminal-bright">RESULTS</h3>
            <div className={`px-3 py-1 rounded text-xs font-mono font-semibold ${
              result.passed
                ? 'bg-terminal-green/10 border border-terminal-green text-terminal-green'
                : 'bg-terminal-red/10 border border-terminal-red text-terminal-red'
            }`}>
              {result.passed ? '✓ PASSED' : '✗ FAILED'}
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div>
              <div className="text-[10px] font-mono text-terminal-text/40 mb-1">SHARPE</div>
              <div className="text-lg font-mono font-bold text-terminal-text metric-value">
                {result.metrics.sharpe_ratio.toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-[10px] font-mono text-terminal-text/40 mb-1">DSR</div>
              <div className="text-lg font-mono font-bold text-terminal-text metric-value">
                {result.metrics.dsr.toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-[10px] font-mono text-terminal-text/40 mb-1">RETURN</div>
              <div className="text-lg font-mono font-bold text-terminal-green metric-value">
                {(result.metrics.annual_return * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-[10px] font-mono text-terminal-text/40 mb-1">DRAWDOWN</div>
              <div className="text-lg font-mono font-bold text-terminal-red metric-value">
                {(result.metrics.max_drawdown * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}
