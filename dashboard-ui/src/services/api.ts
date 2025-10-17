import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000'

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 second timeout
})

// Add request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
    return config
  },
  (error) => {
    console.error('API Request Error:', error)
    return Promise.reject(error)
  }
)

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`)
    return response
  },
  (error) => {
    console.error('API Response Error:', error.response?.status, error.message)
    return Promise.reject(error)
  }
)

export interface DashboardOverview {
  total_anomalies: number
  open_anomalies: number
  acknowledged_anomalies: number
  resolved_anomalies: number
  recent_anomalies: number
  severity_distribution: Record<string, number>
}

export interface TimelineData {
  date: string
  count: number
}

export interface Distributions {
  status_distribution: Record<string, number>
  enterprise_distribution: Array<{
    enterprise_id: string
    count: number
  }>
}

export interface Anomaly {
  id: number
  source_event: string
  resource_id: string
  resource_name: string
  severity: string
  timestamp: string
  enterprise_id: string | null
  status: string
}

export interface AnomaliesResponse {
  anomalies: Anomaly[]
  total: number
  limit: number
  offset: number
  has_more: boolean
}

export interface HealthStatus {
  status: string
  timestamp: string
  database: string
  error?: string
}

// Dashboard endpoints
export const getDashboardOverview = async (): Promise<DashboardOverview> => {
  const response = await api.get('/api/dashboard/overview')
  return response.data
}

export const getTimeline = async (days: number = 7): Promise<TimelineData[]> => {
  const response = await api.get(`/api/dashboard/timeline?days=${days}`)
  return response.data
}

export const getDistributions = async (): Promise<Distributions> => {
  const response = await api.get('/api/dashboard/distributions')
  return response.data
}

// Anomaly endpoints
export const getAllAnomalies = async (params: {
  status?: string
  severity?: string
  enterprise_id?: string
  limit?: number
  offset?: number
} = {}): Promise<AnomaliesResponse> => {
  const response = await api.get('/api/anomalies/all', { params })
  return response.data
}

export const searchAnomalies = async (query: string, limit: number = 20): Promise<Anomaly[]> => {
  const response = await api.get(`/api/anomalies/search?q=${encodeURIComponent(query)}&limit=${limit}`)
  return response.data
}

export const updateAnomalyStatus = async (anomalyId: number, status: string): Promise<Anomaly> => {
  try {
    console.log(`[API] Sending PUT request to /api/anomalies/${anomalyId}/status with status: ${status}`)
    const response = await api.put(`/api/anomalies/${anomalyId}/status`, { status })
    console.log('[API] Response received:', response.data)
    return response.data
  } catch (error: any) {
    console.error('[API] Error updating anomaly status:', {
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      message: error.message
    })
    throw error
  }
}

// Health check
export const getHealthStatus = async (): Promise<HealthStatus> => {
  const response = await api.get('/api/health')
  return response.data
}

export default api
