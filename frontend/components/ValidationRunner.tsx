'use client'

import { motion, AnimatePresence } from 'framer-motion'
import { useState, useEffect } from 'react'
import { validateStrategy, pollValidationResult, listTemplates, type ValidationResult, type Template } from '@/lib/api'

export default function ValidationRunner() {
  const [symbol, setSymbol] = useState('SPY')
  const [template, setTemplate] = useState('sma_crossover')
  const [templates, setTemplates] = useState<Template[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [logs, setLogs] = useState<string[]>([])
  const [result, setResult] = useState<ValidationResult | null>(null)
  const [runCpcv, setRunCpcv] = useState(true)
  const [runStress, setRunStress] = useState(true)

  // Fetch templates on mount
  useEffect(() => {
    const fetchTemplates = async () => {
      try {
        const data = await listTemplates()
        setTemplates(data)
        if (data.length > 0 && !data.find(t => t.name === template)) {
          setTemplate(data[0].name)
        }
      } catch (err) {
        console.error('Failed to fetch templates:', err)
      }
    }
    fetchTemplates()
  }, [])

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
        run_cpcv: runCpcv,
        run_spa: false,
        run_stress: runStress,
      })

      setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Validation started: ${response.validation_id}`])

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
      setLogs(prev => [...prev, `[${new Date().toLocaleTimeString()}] Error: ${error}`])
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
          <div className="w-1 h-6 bg-terminal-green" />
          <h2 className="text-xl font-display font-bold text-terminal-bright tracking-wide">
            SINGLE STRATEGY VALIDATION
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
      <div className="grid grid-cols-2 gap-4 mb-4">
        <div>
          <label className="text-xs font-mono text-terminal-text/60 mb-2 block">SYMBOL</label>
          <input
            type="text"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            disabled={isRunning}
            className="w-full px-3 py-2 bg-terminal-bg border border-terminal-border rounded text-sm font-mono text-terminal-bright focus:outline-none focus:border-terminal-green disabled:opacity-50"
          />
        </div>
        <div>
          <label className="text-xs font-mono text-terminal-text/60 mb-2 block">TEMPLATE</label>
          <select
            value={template}
            onChange={(e) => setTemplate(e.target.value)}
            disabled={isRunning}
            className="w-full px-3 py-2 bg-terminal-bg border border-terminal-border rounded text-sm font-mono text-terminal-bright focus:outline-none focus:border-terminal-green disabled:opacity-50"
          >
            {templates.length === 0 ? (
              <option value="">Loading templates...</option>
            ) : (
              templates.map((t) => (
                <option key={t.name} value={t.name}>
                  {t.display_name}
                </option>
              ))
            )}
          </select>
        </div>
      </div>

      {/* Options */}
      <div className="flex gap-6 mb-4">
        <label className="flex items-center gap-2 text-xs font-mono text-terminal-text/80 cursor-pointer">
          <input
            type="checkbox"
            checked={runCpcv}
            onChange={(e) => setRunCpcv(e.target.checked)}
            disabled={isRunning}
            className="w-4 h-4 rounded border-terminal-border bg-terminal-bg text-terminal-green focus:ring-terminal-green"
          />
          Run CPCV
        </label>
        <label className="flex items-center gap-2 text-xs font-mono text-terminal-text/80 cursor-pointer">
          <input
            type="checkbox"
            checked={runStress}
            onChange={(e) => setRunStress(e.target.checked)}
            disabled={isRunning}
            className="w-4 h-4 rounded border-terminal-border bg-terminal-bg text-terminal-green focus:ring-terminal-green"
          />
          Run Stress Test
        </label>
      </div>

      {/* Run Button */}
      <button
        onClick={handleRunValidation}
        disabled={isRunning || templates.length === 0}
        className="w-full px-4 py-3 bg-terminal-green text-terminal-bg rounded font-mono font-semibold hover:bg-terminal-green/90 transition-all disabled:opacity-50 disabled:cursor-not-allowed mb-6"
      >
        {isRunning ? 'VALIDATING...' : 'START VALIDATION'}
      </button>

      {/* Logs Output */}
      {logs.length > 0 && (
        <div className="bg-terminal-bg border border-terminal-border rounded-lg p-4 max-h-48 overflow-y-auto font-mono text-xs mb-4">
          <AnimatePresence>
            {logs.map((log, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                className={`${
                  log.includes('✓') || log.includes('PASSED') ? 'text-terminal-green' :
                  log.includes('✗') || log.includes('FAIL') || log.includes('Error') ? 'text-terminal-red' :
                  log.includes('⚠') ? 'text-terminal-amber' :
                  'text-terminal-text/80'
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
          className="p-4 border border-terminal-border rounded-lg bg-terminal-bg/50"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-mono font-bold text-terminal-bright">RESULTS</h3>
            <div className={`px-3 py-1 rounded text-xs font-mono font-semibold ${
              result.passed
                ? 'bg-terminal-green/10 border border-terminal-green text-terminal-green'
                : 'bg-terminal-red/10 border border-terminal-red text-terminal-red'
            }`}>
              {result.passed ? 'PASSED' : 'FAILED'}
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div>
              <div className="text-[10px] font-mono text-terminal-text/40 mb-1">SHARPE</div>
              <div className={`text-lg font-mono font-bold metric-value ${result.metrics.sharpe_ratio > 1 ? 'text-terminal-green' : 'text-terminal-amber'}`}>
                {result.metrics.sharpe_ratio.toFixed(2)}
              </div>
            </div>
            <div>
              <div className="text-[10px] font-mono text-terminal-text/40 mb-1">DSR</div>
              <div className={`text-lg font-mono font-bold metric-value ${result.metrics.dsr > 0.95 ? 'text-terminal-green' : 'text-terminal-red'}`}>
                {result.metrics.dsr.toFixed(3)}
              </div>
            </div>
            <div>
              <div className="text-[10px] font-mono text-terminal-text/40 mb-1">RETURN</div>
              <div className={`text-lg font-mono font-bold metric-value ${result.metrics.annual_return > 0 ? 'text-terminal-green' : 'text-terminal-red'}`}>
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
