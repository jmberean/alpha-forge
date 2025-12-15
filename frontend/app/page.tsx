'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import Header from '@/components/Header'
import ValidationPipeline from '@/components/ValidationPipeline'
import MetricsGrid from '@/components/MetricsGrid'
import StrategyList from '@/components/StrategyList'
import ValidationRunner from '@/components/ValidationRunner'
import StrategyFactory from '@/components/StrategyFactory'
import StrategyDiscovery from '@/components/StrategyDiscovery'

export default function Dashboard() {
  const [mounted, setMounted] = useState(false)
  const [activeTab, setActiveTab] = useState<'discovery' | 'factory'>('discovery')

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
              <span className="text-[10px] font-mono text-terminal-cyan bg-terminal-cyan/10 px-2 py-0.5 rounded">
                MVP8
              </span>
            </div>
            <p className="text-terminal-text text-lg font-mono">
              {'>'} Defense-in-depth systematic strategy validation platform
            </p>
            <p className="text-terminal-text/60 text-sm font-mono mt-2">
              Multi-objective genetic programming • Expression tree discovery • NSGA-III optimizer
            </p>
          </div>
        </motion.div>

        {/* Strategy Generation Tabs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <div className="flex gap-4 mb-6">
            <button
              onClick={() => setActiveTab('discovery')}
              className={`
                px-6 py-3 rounded font-mono font-semibold text-sm tracking-wider transition-all duration-300
                ${
                  activeTab === 'discovery'
                    ? 'bg-terminal-cyan text-terminal-bg border-2 border-terminal-cyan'
                    : 'bg-terminal-panel/50 text-terminal-text border-2 border-terminal-border hover:border-terminal-cyan/50'
                }
              `}
            >
              <span className="flex items-center gap-2">
                <span>STRATEGY DISCOVERY</span>
                <span className="text-[10px] px-2 py-0.5 rounded bg-terminal-green/20 text-terminal-green">NEW</span>
              </span>
            </button>
            <button
              onClick={() => setActiveTab('factory')}
              className={`
                px-6 py-3 rounded font-mono font-semibold text-sm tracking-wider transition-all duration-300
                ${
                  activeTab === 'factory'
                    ? 'bg-terminal-cyan text-terminal-bg border-2 border-terminal-cyan'
                    : 'bg-terminal-panel/50 text-terminal-text border-2 border-terminal-border hover:border-terminal-cyan/50'
                }
              `}
            >
              GENETIC FACTORY
            </button>
          </div>

          {/* Tab Content */}
          {activeTab === 'discovery' ? <StrategyDiscovery /> : <StrategyFactory />}
        </motion.div>

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
