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
    pbo?: number
    annual_return: number
    max_drawdown: number
    sortino_ratio: number
    total_return: number
    volatility: number
  }
  timestamp: string
  logs: string[]
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
