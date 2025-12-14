'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Header from '@/components/Header'
import ValidationPipeline from '@/components/ValidationPipeline'
import MetricsGrid from '@/components/MetricsGrid'
import StrategyList from '@/components/StrategyList'
import ValidationRunner from '@/components/ValidationRunner'
import StrategyFactory from '@/components/StrategyFactory'

export default function Dashboard() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) return null

  return (
    <div className="min-h-screen bg-terminal-bg relative z-10">
      <Header />

      <main className="container mx-auto px-6 py-8 space-y-8">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="relative"
        >
          <div className="absolute inset-0 bg-gradient-to-r from-terminal-green/10 to-terminal-amber/10 blur-3xl opacity-30" />
          <div className="relative border border-terminal-border bg-terminal-panel/50 backdrop-blur-sm p-8 rounded-lg">
            <div className="flex items-center gap-4 mb-4">
              <div className="w-2 h-2 bg-terminal-green rounded-full animate-pulse" />
              <h1 className="text-4xl font-display font-bold text-terminal-bright tracking-wider">
                ALPHAFORGE<span className="text-terminal-green">_</span>
              </h1>
            </div>
            <p className="text-terminal-text text-lg font-mono">
              {'>'} Defense-in-depth systematic strategy validation platform
            </p>
            <p className="text-terminal-text/60 text-sm font-mono mt-2">
              Protecting against overfitting, lookahead bias, and execution reality mismatch
            </p>
          </div>
        </motion.div>

        {/* Strategy Factory - Main Feature */}
        <StrategyFactory />

        {/* Validation Pipeline Stats */}
        <ValidationPipeline />

        {/* Metrics Grid */}
        <MetricsGrid />

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Strategy Candidates */}
          <StrategyList />

          {/* Single Strategy Validation */}
          <ValidationRunner />
        </div>
      </main>
    </div>
  )
}
