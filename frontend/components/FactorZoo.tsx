'use client'

import { motion } from 'framer-motion'
import { useState } from 'react'

interface FactorZooProps {
  formulas: string[]
}

export default function FactorZoo({ formulas }: FactorZooProps) {
  const [searchTerm, setSearchTerm] = useState('')
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null)

  const filteredFormulas = formulas.filter((formula) =>
    formula.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const handleCopy = (formula: string, index: number) => {
    navigator.clipboard.writeText(formula)
    setCopiedIndex(index)
    setTimeout(() => setCopiedIndex(null), 2000)
  }

  const getOperatorCount = (formula: string): number => {
    return (formula.match(/\w+\(/g) || []).length
  }

  const extractOperators = (formula: string): string[] => {
    const operators = formula.match(/\w+(?=\()/g) || []
    return Array.from(new Set(operators))
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.4 }}
      className="border border-terminal-cyan bg-terminal-panel/50 backdrop-blur-sm rounded-lg p-6"
    >
      <div className="flex items-center gap-3 mb-6">
        <div className="w-1 h-6 bg-terminal-cyan" />
        <h3 className="text-lg font-display font-bold text-terminal-bright">FACTOR ZOO</h3>
        <span className="text-[10px] font-mono text-terminal-text/60">
          {formulas.length} Validated Formulas
        </span>
      </div>

      {/* Search */}
      <div className="mb-6">
        <label className="block text-[10px] font-mono text-terminal-text/60 mb-2">SEARCH FORMULAS</label>
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Search by operator or pattern..."
          className="w-full bg-terminal-bg border border-terminal-border rounded px-3 py-2 text-sm font-mono text-terminal-bright placeholder:text-terminal-text/40 focus:border-terminal-cyan focus:outline-none"
        />
      </div>

      {/* Info Box */}
      <div className="mb-6 p-3 bg-terminal-cyan/5 border border-terminal-cyan/20 rounded">
        <div className="text-[10px] font-mono text-terminal-text/60 mb-1">ABOUT FACTOR ZOO</div>
        <div className="text-xs font-mono text-terminal-text">
          High-quality formulas that passed validation on held-out data. These strategies demonstrated Sharpe &gt; {' '}
          threshold and low complexity, making them suitable for production deployment.
        </div>
      </div>

      {/* Formula List */}
      <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
        {filteredFormulas.length === 0 ? (
          <div className="text-center py-12 text-terminal-text/60 font-mono text-sm">
            {searchTerm ? 'No formulas match your search' : 'No formulas in zoo yet'}
          </div>
        ) : (
          filteredFormulas.map((formula, idx) => {
            const operators = extractOperators(formula)
            const operatorCount = getOperatorCount(formula)

            return (
              <motion.div
                key={idx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: idx * 0.05 }}
                className="bg-terminal-bg border border-terminal-border rounded p-4 hover:border-terminal-cyan transition-colors"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <div className="text-sm font-mono font-semibold text-terminal-cyan mb-1">
                      Formula #{idx + 1}
                    </div>
                    <div className="text-xs font-mono text-terminal-text/60">
                      {operatorCount} operators • {formula.length} chars
                    </div>
                  </div>
                  <button
                    onClick={() => handleCopy(formula, idx)}
                    className={`
                      px-3 py-1 rounded text-[10px] font-mono font-semibold transition-all
                      ${
                        copiedIndex === idx
                          ? 'bg-terminal-green/20 text-terminal-green border border-terminal-green'
                          : 'bg-terminal-cyan/10 text-terminal-cyan border border-terminal-cyan/30 hover:bg-terminal-cyan/20'
                      }
                    `}
                  >
                    {copiedIndex === idx ? '✓ COPIED' : 'COPY'}
                  </button>
                </div>

                {/* Formula */}
                <div className="bg-terminal-bg/50 border border-terminal-border/50 rounded p-3 mb-3 overflow-x-auto">
                  <code className="text-xs font-mono text-terminal-bright whitespace-nowrap">
                    {formula}
                  </code>
                </div>

                {/* Operators Used */}
                <div>
                  <div className="text-[10px] font-mono text-terminal-text/60 mb-2">OPERATORS USED</div>
                  <div className="flex flex-wrap gap-2">
                    {operators.map((op, opIdx) => (
                      <span
                        key={opIdx}
                        className="px-2 py-1 rounded text-[10px] font-mono bg-terminal-cyan/10 text-terminal-cyan border border-terminal-cyan/30"
                      >
                        {op}
                      </span>
                    ))}
                  </div>
                </div>
              </motion.div>
            )
          })
        )}
      </div>

      {/* Summary Stats */}
      {filteredFormulas.length > 0 && (
        <div className="mt-6 grid grid-cols-3 gap-4">
          <div className="bg-terminal-bg border border-terminal-border rounded p-3 text-center">
            <div className="text-lg font-mono font-bold text-terminal-cyan">{filteredFormulas.length}</div>
            <div className="text-[10px] font-mono text-terminal-text/60">TOTAL FORMULAS</div>
          </div>
          <div className="bg-terminal-bg border border-terminal-border rounded p-3 text-center">
            <div className="text-lg font-mono font-bold text-terminal-amber">
              {Math.round(
                filteredFormulas.reduce((sum, f) => sum + getOperatorCount(f), 0) / filteredFormulas.length
              )}
            </div>
            <div className="text-[10px] font-mono text-terminal-text/60">AVG OPERATORS</div>
          </div>
          <div className="bg-terminal-bg border border-terminal-border rounded p-3 text-center">
            <div className="text-lg font-mono font-bold text-terminal-green">
              {Array.from(new Set(filteredFormulas.flatMap(extractOperators))).length}
            </div>
            <div className="text-[10px] font-mono text-terminal-text/60">UNIQUE OPERATORS</div>
          </div>
        </div>
      )}
    </motion.div>
  )
}
