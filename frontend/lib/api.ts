/**
 * AlphaForge API Client
 *
 * Connects to the FastAPI backend to run validations and fetch results.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface ValidationRequest {
  symbol: string
  template: string
  start_date?: string
  end_date?: string
  n_trials?: number
  run_cpcv?: boolean
  run_spa?: boolean
  run_stress?: boolean
}

export interface ValidationResponse {
  validation_id: string
  status: string
  message: string
}

export interface ValidationResult {
  validation_id: string
  status: 'running' | 'completed' | 'failed'
  strategy_name: string
  passed: boolean
  metrics: {
    sharpe_ratio: number
    dsr: number
    prob_loss?: number
    annual_return: number
    max_drawdown: number
    sortino_ratio: number
    total_return: number
    volatility: number
    num_trades?: number
    win_rate?: number
    profit_factor?: number
    avg_win?: number
    avg_loss?: number
  }
  timestamp: string
  logs: string[]
  equity_curve?: {
    date: string
    equity: number
    benchmark: number
  }[]
}

export interface Strategy {
  id: string
  name: string
  type: string
  status: string
  sharpe: number
  dsr: number
  annual_return: number
}

export interface Template {
  name: string
  display_name: string
}

export interface FactoryRequest {
  symbol: string
  start_date?: string
  end_date?: string
  population_size?: number
  generations?: number
  target_strategies?: number
  validate_top_n?: number
}

export interface FactoryResponse {
  factory_id: string
  status: string
  message: string
}

export interface FactoryResult {
  factory_id: string
  status: 'running' | 'completed' | 'failed'
  strategies: Strategy[]
  stats: {
    generated: number
    validated: number
    passed: number
  }
  timestamp: string
  logs: string[]
}

export interface PipelineStats {
  stages: {
    name: string
    count: number
    rate: number
  }[]
  totals: {
    total_generated: number
    total_validated: number
    total_passed: number
    total_deployed: number
  }
}

export interface MetricsData {
  has_data: boolean
  strategy_name?: string
  passed?: boolean
  timestamp?: string
  metrics: {
    name: string
    value: number
    threshold: string
    unit: string
  }[]
}

export interface SystemStatus {
  status: string
  version: string
  total_validations: number
  passed_validations: number
  pass_rate: number
  factory_runs: number
  discovery_runs: number
}

export interface DiscoveryRequest {
  symbol: string
  start_date?: string
  end_date?: string
  population_size?: number
  n_generations?: number
  n_objectives?: number
  min_sharpe?: number
  max_turnover?: number
  max_complexity?: number
  validation_split?: number
}

export interface DiscoveryResponse {
  discovery_id: string
  status: string
  message: string
}

export interface ExpressionStrategy {
  formula: string
  size: number
  depth: number
  complexity: number
  fitness: {
    sharpe: number
    drawdown: number
    turnover: number
    complexity: number
  }
}

export interface GenerationStats {
  generation: number
  pareto_front_size: number
  fitness: {
    avg: Record<string, number>
    best: Record<string, number>
  }
  diversity: {
    unique_formulas: number
    avg_size: number
    avg_depth: number
  }
}

export interface DiscoveryResult {
  discovery_id: string
  status: 'running' | 'completed' | 'failed'
  pareto_front: ExpressionStrategy[]
  factor_zoo: string[]
  best_by_objective: Record<string, ExpressionStrategy>
  ensemble_weights: Record<string, number>
  generation_stats: GenerationStats[]
  timestamp: string
  logs: string[]
  error?: string
}

/**
 * Start a new strategy validation
 */
export async function validateStrategy(request: ValidationRequest): Promise<ValidationResponse> {
  const response = await fetch(`${API_BASE}/api/validate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    throw new Error(`Validation failed: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Get validation result by ID
 */
export async function getValidationResult(validationId: string): Promise<ValidationResult> {
  const response = await fetch(`${API_BASE}/api/validate/${validationId}`)

  if (!response.ok) {
    throw new Error(`Failed to fetch validation: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Poll for validation result until complete
 */
export async function pollValidationResult(
  validationId: string,
  onUpdate?: (result: ValidationResult) => void,
  intervalMs: number = 1000
): Promise<ValidationResult> {
  return new Promise((resolve, reject) => {
    const interval = setInterval(async () => {
      try {
        const result = await getValidationResult(validationId)

        if (onUpdate) {
          onUpdate(result)
        }

        if (result.status === 'completed' || result.status === 'failed') {
          clearInterval(interval)
          resolve(result)
        }
      } catch (error) {
        clearInterval(interval)
        reject(error)
      }
    }, intervalMs)
  })
}

/**
 * List all validated strategies
 */
export async function listStrategies(): Promise<Strategy[]> {
  const response = await fetch(`${API_BASE}/api/strategies`)

  if (!response.ok) {
    throw new Error(`Failed to fetch strategies: ${response.statusText}`)
  }

  return response.json()
}

/**
 * List available strategy templates
 */
export async function listTemplates(): Promise<Template[]> {
  const response = await fetch(`${API_BASE}/api/templates`)

  if (!response.ok) {
    throw new Error(`Failed to fetch templates: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Start a strategy factory run
 */
export async function runFactory(request: FactoryRequest): Promise<FactoryResponse> {
  const response = await fetch(`${API_BASE}/api/factory`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    throw new Error(`Factory failed: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Get factory result by ID
 */
export async function getFactoryResult(factoryId: string): Promise<FactoryResult> {
  const response = await fetch(`${API_BASE}/api/factory/${factoryId}`)

  if (!response.ok) {
    throw new Error(`Failed to fetch factory result: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Poll for factory result until complete
 */
export async function pollFactoryResult(
  factoryId: string,
  onUpdate?: (result: FactoryResult) => void,
  intervalMs: number = 2000
): Promise<FactoryResult> {
  return new Promise((resolve, reject) => {
    const interval = setInterval(async () => {
      try {
        const result = await getFactoryResult(factoryId)

        if (onUpdate) {
          onUpdate(result)
        }

        if (result.status === 'completed' || result.status === 'failed') {
          clearInterval(interval)
          resolve(result)
        }
      } catch (error) {
        clearInterval(interval)
        reject(error)
      }
    }, intervalMs)
  })
}

/**
 * Get pipeline statistics
 */
export async function getPipelineStats(): Promise<PipelineStats> {
  const response = await fetch(`${API_BASE}/api/pipeline-stats`)

  if (!response.ok) {
    throw new Error(`Failed to fetch pipeline stats: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Get latest metrics
 */
export async function getLatestMetrics(): Promise<MetricsData> {
  const response = await fetch(`${API_BASE}/api/metrics/latest`)

  if (!response.ok) {
    throw new Error(`Failed to fetch metrics: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Get system status
 */
export async function getSystemStatus(): Promise<SystemStatus> {
  const response = await fetch(`${API_BASE}/api/system/status`)

  if (!response.ok) {
    throw new Error(`Failed to fetch system status: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Start a new discovery run
 */
export async function runDiscovery(request: DiscoveryRequest): Promise<DiscoveryResponse> {
  const response = await fetch(`${API_BASE}/api/discovery`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    throw new Error(`Discovery failed: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Get discovery result by ID
 */
export async function getDiscoveryResult(discoveryId: string): Promise<DiscoveryResult> {
  const response = await fetch(`${API_BASE}/api/discovery/${discoveryId}`)

  if (!response.ok) {
    throw new Error(`Failed to fetch discovery result: ${response.statusText}`)
  }

  return response.json()
}

/**
 * Poll for discovery result until complete
 */
export async function pollDiscoveryResult(
  discoveryId: string,
  onUpdate?: (result: DiscoveryResult) => void,
  intervalMs: number = 2000
): Promise<DiscoveryResult> {
  return new Promise((resolve, reject) => {
    const interval = setInterval(async () => {
      try {
        const result = await getDiscoveryResult(discoveryId)

        if (onUpdate) {
          onUpdate(result)
        }

        if (result.status === 'completed' || result.status === 'failed') {
          clearInterval(interval)
          resolve(result)
        }
      } catch (error) {
        clearInterval(interval)
        reject(error)
      }
    }, intervalMs)
  })
}
