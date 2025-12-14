'use client'

import { motion } from 'framer-motion'

const pipelineStages = [
  { name: 'Generation', count: 10000, survivors: 10000, pass_rate: 100, color: 'text-terminal-blue' },
  { name: 'DSR Screen', count: 10000, survivors: 450, pass_rate: 4.5, color: 'text-terminal-cyan', threshold: 'DSR > 0.95' },
  { name: 'CPCV/PBO', count: 450, survivors: 89, pass_rate: 19.8, color: 'text-terminal-purple', threshold: 'PBO < 0.05' },
  { name: 'Event-Driven', count: 89, survivors: 23, pass_rate: 25.8, color: 'text-terminal-amber', threshold: 'Shortfall < 30%' },
  { name: 'SPA + Stress', count: 23, survivors: 8, pass_rate: 34.8, color: 'text-terminal-green', threshold: 'p < 0.05, 80% pass' },
  { name: 'Shadow Trading', count: 8, survivors: 3, pass_rate: 37.5, color: 'text-terminal-green', threshold: 'Live validation' },
  { name: 'DEPLOYED', count: 3, survivors: 3, pass_rate: 100, color: 'text-terminal-green', threshold: '' },
]

export default function ValidationPipeline() {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6, delay: 0.2 }}
      className="border border-terminal-border bg-terminal-panel/50 backdrop-blur-sm rounded-lg p-8"
    >
      <div className="flex items-center gap-3 mb-6">
        <div className="w-1 h-6 bg-terminal-green" />
        <h2 className="text-2xl font-display font-bold text-terminal-bright tracking-wide">
          VALIDATION PIPELINE
        </h2>
        <div className="text-xs font-mono text-terminal-text/60 ml-auto">
          SURVIVAL RATE: <span className="text-terminal-amber metric-value">0.03%</span>
        </div>
      </div>

      <div className="relative">
        {/* Pipeline Stages */}
        <div className="grid grid-cols-1 md:grid-cols-7 gap-4">
          {pipelineStages.map((stage, idx) => (
            <motion.div
              key={stage.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.3 + idx * 0.1 }}
              className="relative group"
            >
              {/* Connection Line */}
              {idx < pipelineStages.length - 1 && (
                <div className="hidden md:block absolute top-1/2 -right-2 w-4 h-0.5 bg-terminal-border z-0">
                  <div
                    className="h-full bg-gradient-to-r from-terminal-green to-transparent animate-pulse-slow"
                    style={{
                      width: `${(stage.survivors / stage.count) * 100}%`,
                    }}
                  />
                </div>
              )}

              {/* Stage Card */}
              <div
                className={`
                  relative border border-terminal-border bg-terminal-bg/80 p-4 rounded
                  hover:border-terminal-green transition-all duration-300
                  ${idx === pipelineStages.length - 1 ? 'animate-glow' : ''}
                `}
              >
                {/* Stage Name */}
                <div className="text-xs font-mono text-terminal-text/60 mb-2">
                  STAGE {idx + 1}
                </div>
                <div className={`text-sm font-display font-bold ${stage.color} mb-3`}>
                  {stage.name.toUpperCase()}
                </div>

                {/* Counts */}
                <div className="space-y-1 mb-3">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-terminal-text/60">In:</span>
                    <span className="text-terminal-text metric-value">{stage.count.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-terminal-text/60">Out:</span>
                    <span className={`${stage.color} metric-value font-semibold`}>
                      {stage.survivors.toLocaleString()}
                    </span>
                  </div>
                </div>

                {/* Pass Rate Bar */}
                <div className="w-full h-1.5 bg-terminal-border rounded-full overflow-hidden mb-2">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(stage.pass_rate, 100)}%` }}
                    transition={{ duration: 1, delay: 0.5 + idx * 0.1 }}
                    className="h-full bg-gradient-to-r from-terminal-red via-terminal-amber to-terminal-green"
                  />
                </div>
                <div className="text-xs font-mono text-terminal-text/60 text-right">
                  {stage.pass_rate.toFixed(1)}%
                </div>

                {/* Threshold */}
                {stage.threshold && (
                  <div className="mt-3 pt-3 border-t border-terminal-border/50">
                    <div className="text-[10px] font-mono text-terminal-text/40">
                      {stage.threshold}
                    </div>
                  </div>
                )}

                {/* Deployment Indicator */}
                {idx === pipelineStages.length - 1 && (
                  <div className="absolute -top-2 -right-2 w-4 h-4 bg-terminal-green rounded-full animate-pulse">
                    <div className="absolute inset-0 bg-terminal-green rounded-full animate-ping" />
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Statistics Footer */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
          className="mt-8 pt-6 border-t border-terminal-border"
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-xs font-mono text-terminal-text/60 mb-1">TOTAL TESTED</div>
              <div className="text-2xl font-mono font-bold text-terminal-blue metric-value">10,000</div>
            </div>
            <div>
              <div className="text-xs font-mono text-terminal-text/60 mb-1">DEPLOYED</div>
              <div className="text-2xl font-mono font-bold text-terminal-green metric-value">3</div>
            </div>
            <div>
              <div className="text-xs font-mono text-terminal-text/60 mb-1">AVG TIME</div>
              <div className="text-2xl font-mono font-bold text-terminal-amber metric-value">47m</div>
            </div>
            <div>
              <div className="text-xs font-mono text-terminal-text/60 mb-1">FALSE DISCOVERY</div>
              <div className="text-2xl font-mono font-bold text-terminal-red metric-value">&lt;5%</div>
            </div>
          </div>
        </motion.div>
      </div>
    </motion.div>
  )
}
