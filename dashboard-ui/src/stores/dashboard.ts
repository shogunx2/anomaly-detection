import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { DashboardOverview, TimelineData, Distributions, Anomaly, AnomaliesResponse } from '@/services/api'
import { getDashboardOverview, getTimeline, getDistributions, getAllAnomalies, searchAnomalies, getHealthStatus, updateAnomalyStatus } from '@/services/api'

export interface Notification {
  id: number
  anomalyId: number
  message: string
  severity: string
  timestamp: Date
  resourceName: string
}

export const useDashboardStore = defineStore('dashboard', () => {
  // State
  const overview = ref<DashboardOverview | null>(null)
  const timeline = ref<TimelineData[]>([])
  const distributions = ref<Distributions | null>(null)
  const anomalies = ref<Anomaly[]>([])
  const searchResults = ref<Anomaly[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  const isConnected = ref(false)
  const lastHealthCheck = ref<Date | null>(null)
  const notifications = ref<Notification[]>([])
  const lastCheckedAnomalyId = ref<number>(0)
  let notificationCheckInterval: ReturnType<typeof setInterval> | null = null

  // Computed
  const totalAnomalies = computed(() => overview.value?.total_anomalies || 0)
  const openAnomalies = computed(() => overview.value?.open_anomalies || 0)
  const acknowledgedAnomalies = computed(() => overview.value?.acknowledged_anomalies || 0)
  const resolvedAnomalies = computed(() => overview.value?.resolved_anomalies || 0)
  const recentAnomalies = computed(() => overview.value?.recent_anomalies || 0)

  // Actions
  const fetchOverview = async () => {
    console.debug('[Store.fetchOverview] Starting fetch')
    loading.value = true
    error.value = null
    try {
      const data = await getDashboardOverview()
      console.debug('[Store.fetchOverview] Data received:', {
        total_anomalies: data.total_anomalies,
        open_anomalies: data.open_anomalies,
        acknowledged_anomalies: data.acknowledged_anomalies,
        resolved_anomalies: data.resolved_anomalies,
        recent_anomalies: data.recent_anomalies,
        severity_distribution: data.severity_distribution
      })
      overview.value = data
    } catch (err) {
      error.value = 'Failed to fetch dashboard overview'
      console.error('[Store.fetchOverview] Error:', err)
      // Set default values to prevent UI issues
      overview.value = {
        total_anomalies: 0,
        open_anomalies: 0,
        acknowledged_anomalies: 0,
        resolved_anomalies: 0,
        recent_anomalies: 0,
        severity_distribution: {}
      }
    } finally {
      loading.value = false
    }
  }

  const fetchTimeline = async (days: number = 7) => {
    console.debug('[Store.fetchTimeline] Starting fetch with days:', days)
    loading.value = true
    error.value = null
    try {
      const data = await getTimeline(days)
      console.debug('[Store.fetchTimeline] Data received:', {
        length: data.length,
        sample: data.slice(0, 3),
        dates: data.map(d => d.date),
        counts: data.map(d => d.count)
      })
      timeline.value = data
    } catch (err) {
      error.value = 'Failed to fetch timeline data'
      console.error('[Store.fetchTimeline] Error:', err)
    } finally {
      loading.value = false
    }
  }

  const fetchDistributions = async () => {
    console.debug('[Store.fetchDistributions] Starting fetch')
    loading.value = true
    error.value = null
    try {
      const data = await getDistributions()
      console.debug('[Store.fetchDistributions] Data received:', {
        status_distribution: data.status_distribution,
        enterprise_distribution_count: data.enterprise_distribution.length
      })
      distributions.value = data
    } catch (err) {
      error.value = 'Failed to fetch distribution data'
      console.error('[Store.fetchDistributions] Error:', err)
    } finally {
      loading.value = false
    }
  }

  const fetchAnomalies = async (params: {
    status?: string
    severity?: string
    enterprise_id?: string
    limit?: number
    offset?: number
  } = {}) => {
    loading.value = true
    error.value = null
    try {
      const response = await getAllAnomalies(params)
      anomalies.value = response.anomalies
      return response // Return the full response for pagination info
    } catch (err) {
      error.value = 'Failed to fetch anomalies'
      console.error('Error fetching anomalies:', err)
      return { anomalies: [], total: 0, has_more: false }
    } finally {
      loading.value = false
    }
  }

  const searchAnomaliesQuery = async (query: string, limit: number = 20) => {
    loading.value = true
    error.value = null
    try {
      searchResults.value = await searchAnomalies(query, limit)
    } catch (err) {
      error.value = 'Failed to search anomalies'
      console.error('Error searching anomalies:', err)
    } finally {
      loading.value = false
    }
  }

  const checkHealth = async () => {
    console.debug('[Store.checkHealth] Starting health check')
    try {
      const health = await getHealthStatus()
      console.debug('[Store.checkHealth] Health response:', {
        status: health.status,
        database: health.database,
        timestamp: health.timestamp
      })
      isConnected.value = health.status === 'healthy'
      lastHealthCheck.value = new Date()
      
      return isConnected.value
    } catch (err) {
      isConnected.value = false
      lastHealthCheck.value = new Date()
      console.error('[Store.checkHealth] Health check failed:', err)
      return false
    }
  }

  const checkForNewHighSeverityAnomalies = async () => {
    console.debug('[Store.checkForNewHighSeverityAnomalies] Polling for high-severity anomalies')
    try {
      const response = await getAllAnomalies({ severity: 'High', limit: 100 })
      const highSeverityAnomalies = response.anomalies

      // Find new anomalies that weren't in the last check
      highSeverityAnomalies.forEach((anomaly) => {
        if (anomaly.id > lastCheckedAnomalyId.value) {
          console.debug('[Store.checkForNewHighSeverityAnomalies] New high-severity anomaly detected:', {
            id: anomaly.id,
            resource: anomaly.resource_name,
            severity: anomaly.severity
          })

          // Create notification
          const notif: Notification = {
            id: Date.now(),
            anomalyId: anomaly.id,
            message: `New high-severity anomaly: ${anomaly.resource_name}`,
            severity: anomaly.severity,
            timestamp: new Date(),
            resourceName: anomaly.resource_name
          }
          notifications.value.push(notif)
        }
      })

      // Update the last checked anomaly ID
      if (highSeverityAnomalies.length > 0) {
        lastCheckedAnomalyId.value = Math.max(...highSeverityAnomalies.map(a => a.id))
      }
    } catch (err) {
      console.error('[Store.checkForNewHighSeverityAnomalies] Error checking for new anomalies:', err)
    }
  }

  const clearNotifications = () => {
    console.debug('[Store.clearNotifications] Clearing all notifications')
    notifications.value = []
  }

  const startNotificationPolling = () => {
    console.debug('[Store.startNotificationPolling] Starting notification polling (1 min interval)')
    if (notificationCheckInterval) {
      clearInterval(notificationCheckInterval)
    }
    
    // Initial check
    checkForNewHighSeverityAnomalies()
    
    // Poll every 1 minute (60000 ms)
    notificationCheckInterval = setInterval(() => {
      checkForNewHighSeverityAnomalies()
    }, 60000)
  }

  const stopNotificationPolling = () => {
    console.debug('[Store.stopNotificationPolling] Stopping notification polling')
    if (notificationCheckInterval) {
      clearInterval(notificationCheckInterval)
      notificationCheckInterval = null
    }
  }

  const refreshAll = async () => {
    console.debug('[Store.refreshAll] Starting refresh of all data')
    // Perform health check but don't block data fetching strictly on its result.
    // This ensures the Overview still attempts to load data if the API is partially available.
    try {
      await checkHealth()
    } catch (err) {
      console.warn('[Store.refreshAll] Health check failed but proceeding with data fetch:', err)
    }

    console.debug('[Store.refreshAll] Fetching all dashboard data')
    await Promise.all([
      fetchOverview(),
      fetchTimeline(),
      fetchDistributions(),
      fetchAnomalies()
    ])
    console.debug('[Store.refreshAll] All data fetch completed')
  }

  const updateAnomalyStatusAction = async (anomalyId: number, newStatus: string) => {
    console.debug('[Store.updateAnomalyStatusAction] Updating anomaly status', { anomalyId, newStatus })
    try {
      const updatedAnomaly = await updateAnomalyStatus(anomalyId, newStatus)
      console.debug('[Store.updateAnomalyStatusAction] Status updated successfully:', updatedAnomaly)
      
      // Update the anomaly in local state
      const index = anomalies.value.findIndex(a => a.id === anomalyId)
      if (index !== -1) {
        anomalies.value[index] = updatedAnomaly
      }
      
      // Update in search results if present
      const searchIndex = searchResults.value.findIndex(a => a.id === anomalyId)
      if (searchIndex !== -1) {
        searchResults.value[searchIndex] = updatedAnomaly
      }
      
      // Refresh metrics to update overview
      await Promise.all([
        fetchOverview(),
        fetchDistributions()
      ])
      console.debug('[Store.updateAnomalyStatusAction] Metrics refreshed after status update')
      
      return updatedAnomaly
    } catch (err) {
      error.value = 'Failed to update anomaly status'
      console.error('[Store.updateAnomalyStatusAction] Error:', err)
      throw err
    }
  }

  return {
    // State
    overview,
    timeline,
    distributions,
    anomalies,
    searchResults,
    loading,
    error,
    isConnected,
    lastHealthCheck,
    notifications,
    
    // Computed
    totalAnomalies,
    openAnomalies,
    acknowledgedAnomalies,
    resolvedAnomalies,
    recentAnomalies,
    
    // Actions
    fetchOverview,
    fetchTimeline,
    fetchDistributions,
    fetchAnomalies,
    searchAnomaliesQuery,
    checkHealth,
    checkForNewHighSeverityAnomalies,
    clearNotifications,
    startNotificationPolling,
    stopNotificationPolling,
    refreshAll,
    updateAnomalyStatusAction
  }
})
