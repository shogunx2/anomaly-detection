<template>
  <div class="space-y-4">
    <!-- Page Header -->
    <div class="section-header">
      <h1 class="section-title">Anomaly Table</h1>
      <p class="section-subtitle">Complete list of all anomalies with filtering and search</p>
    </div>

    <!-- Filters and Search -->
    <div class="card">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
        <!-- Search -->
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1.5">Search</label>
          <div class="relative">
            <Search class="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-500 w-4 h-4" />
            <input
              v-model="searchQuery"
              @input="debouncedSearch"
              type="text"
              placeholder="Search by resource name..."
              class="input-field pl-10 w-full text-sm py-1.5"
            />
          </div>
        </div>

        <!-- Status Filter -->
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1.5">Status</label>
          <select v-model="filters.status" @change="applyFilters" class="input-field w-full text-sm py-1.5">
            <option value="">All Statuses</option>
            <option value="Open">Open</option>
            <option value="Acknowledged">Acknowledged</option>
            <option value="Resolved">Resolved</option>
          </select>
        </div>

        <!-- Severity Filter -->
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1.5">Severity</label>
          <select v-model="filters.severity" @change="applyFilters" class="input-field w-full text-sm py-1.5">
            <option value="">All Severities</option>
            <option value="High">High</option>
            <option value="Medium">Medium</option>
            <option value="Low">Low</option>
          </select>
        </div>
      </div>

      <div class="flex justify-between items-center pt-3 border-t border-slate-700">
        <button @click="clearFilters" class="btn-secondary text-sm">
          Clear Filters
        </button>
        <div class="text-sm text-slate-400 font-medium">
          Showing <span class="text-slate-300">{{ displayedAnomalies.length }}</span> anomalies
        </div>
      </div>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="flex justify-center items-center h-64">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="bg-danger-500/10 border border-danger-500/30 rounded-lg p-6">
      <p class="text-danger-400 font-medium">Error loading anomalies</p>
      <p class="text-danger-300 text-sm mt-1">{{ error }}</p>
      <button @click="loadAnomalies" class="btn-primary mt-4">Retry</button>
    </div>

    <!-- Table -->
    <div v-else class="card overflow-hidden overflow-x-auto">
      <div class="min-w-max">
        <table class="w-full text-sm">
          <thead>
            <tr class="border-b border-slate-700 bg-slate-800/50">
              <th class="table-header text-left px-3 py-2">
                ID
              </th>
              <th class="table-header text-left px-3 py-2">
                Resource
              </th>
              <th class="table-header text-left px-3 py-2">
                Severity
              </th>
              <th class="table-header text-left px-3 py-2">
                Status
              </th>
              <th class="table-header text-left px-3 py-2">
                Timestamp
              </th>
            </tr>
          </thead>
          <tbody>
            <tr 
              v-for="anomaly in displayedAnomalies" 
              :key="anomaly.id"
              class="table-row"
            >
              <td class="px-3 py-2 text-slate-300 whitespace-nowrap">
                {{ anomaly.id }}
              </td>
              <td class="px-3 py-2">
                <div class="font-medium text-slate-200">{{ anomaly.resource_name }}</div>
                <div class="text-xs text-slate-500">{{ anomaly.resource_id }}</div>
              </td>
              <td class="px-3 py-2 whitespace-nowrap">
                <span 
                  class="badge"
                  :class="getSeverityClass(anomaly.severity)"
                >
                  {{ anomaly.severity }}
                </span>
              </td>
              <td class="px-3 py-2 whitespace-nowrap">
                <div class="relative inline-block">
                  <button
                    @click="toggleStatusDropdown(anomaly.id)"
                    class="badge"
                    :class="getStatusClass(anomaly.status)"
                  >
                    {{ anomaly.status }} ▼
                  </button>
                  
                  <!-- Status Dropdown Menu -->
                  <div 
                    v-if="openStatusDropdown === anomaly.id"
                    class="absolute z-50 mt-2 w-40 bg-slate-900 border border-slate-700 rounded-lg shadow-lg"
                  >
                    <button 
                      v-for="status in ['Open', 'Acknowledged', 'Resolved']"
                      :key="status"
                      @click="updateStatus(anomaly.id, status)"
                      :disabled="status === anomaly.status || updatingStatusId === anomaly.id"
                      class="block w-full text-left px-4 py-2.5 text-sm text-slate-300 hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed first:rounded-t-lg last:rounded-b-lg border-b border-slate-800 last:border-b-0 transition"
                    >
                      <div v-if="updatingStatusId === anomaly.id && updateStatusTarget === status" class="flex items-center">
                        <div class="animate-spin rounded-full h-3 w-3 border-b-2 border-primary-500 mr-2"></div>
                        {{ status }}
                      </div>
                      <div v-else>{{ status }}</div>
                    </button>
                  </div>
                </div>
              </td>
              <td class="px-3 py-2 text-slate-400 whitespace-nowrap">
                {{ formatTimestamp(anomaly.timestamp) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      <!-- Pagination -->
      <div v-if="hasMore || currentPage > 1" class="flex justify-between items-center mt-4 px-4 py-3 bg-slate-800/50 border-t border-slate-700 rounded-b-lg">
        <button 
          @click="previousPage"
          :disabled="currentPage === 1"
          class="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed text-sm"
        >
          Previous
        </button>
        
        <span class="text-sm text-slate-400 font-medium">
          Page <span class="text-slate-300">{{ currentPage }}</span> of <span class="text-slate-300">{{ totalPages }}</span>
        </span>
        
        <button 
          @click="nextPage"
          :disabled="!hasMore"
          class="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed text-sm"
        >
          Next
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { storeToRefs } from 'pinia'
import { useDashboardStore } from '@/stores/dashboard'
import { Search } from 'lucide-vue-next'

const dashboardStore = useDashboardStore()

// Use storeToRefs to preserve reactivity when extracting refs from the Pinia store
const { anomalies, searchResults, loading, error } = storeToRefs(dashboardStore)
const { fetchAnomalies, searchAnomaliesQuery } = dashboardStore

// Reactive state
const searchQuery = ref('')
const currentPage = ref(1)
const pageSize = ref(50)
const totalCount = ref(0)
const hasMore = ref(false)
const isSearching = ref(false)
const openStatusDropdown = ref<number | null>(null)
const updatingStatusId = ref<number | null>(null)
const updateStatusTarget = ref<string | null>(null)

const filters = ref({
  status: '',
  severity: ''
})

// Computed
const totalPages = computed(() => Math.ceil(totalCount.value / pageSize.value))

// Display either search results or filtered anomalies
const displayedAnomalies = computed(() => {
  if (isSearching.value) {
    return searchResults.value
  }
  return anomalies.value
})

// Methods
const getSeverityClass = (severity: string) => {
  switch (severity) {
    case 'High': return 'badge-critical'
    case 'Medium': return 'badge-warning'
    case 'Low': return 'badge-success'
    default: return 'badge-success'
  }
}

const getStatusClass = (status: string) => {
  switch (status) {
    case 'Open': return 'badge-danger'
    case 'Acknowledged': return 'badge-warning'
    case 'Resolved': return 'badge-success'
    default: return 'bg-slate-700 text-slate-300'
  }
}

const formatTimestamp = (timestamp: string) => {
  return new Date(timestamp).toLocaleString()
}

const loadAnomalies = async () => {
  const offset = (currentPage.value - 1) * pageSize.value
  const params = {
    ...filters.value,
    limit: pageSize.value,
    offset
  }
  
  try {
    const response = await fetchAnomalies(params)
    // The store should handle the response, but we need to get the total count
    totalCount.value = response.total || 0
    hasMore.value = response.has_more || false
  } catch (error) {
    console.error('Failed to load anomalies:', error)
  }
}

const applyFilters = () => {
  currentPage.value = 1
  loadAnomalies()
}

const clearFilters = () => {
  filters.value = {
    status: '',
    severity: ''
  }
  searchQuery.value = ''
  currentPage.value = 1
  loadAnomalies()
}

const nextPage = () => {
  if (hasMore.value) {
    currentPage.value++
    loadAnomalies()
  }
}

const previousPage = () => {
  if (currentPage.value > 1) {
    currentPage.value--
    loadAnomalies()
  }
}

const toggleStatusDropdown = (id: number) => {
  openStatusDropdown.value = openStatusDropdown.value === id ? null : id
}

const updateStatus = async (anomalyId: number, newStatus: string) => {
  console.log('[AnomalyTable] Updating anomaly', anomalyId, 'to status', newStatus)
  updatingStatusId.value = anomalyId
  updateStatusTarget.value = newStatus
  
  try {
    await dashboardStore.updateAnomalyStatusAction(anomalyId, newStatus)
    console.log('[AnomalyTable] Status updated successfully')
    openStatusDropdown.value = null
  } catch (err) {
    console.error('[AnomalyTable] Failed to update status:', err)
    alert('Failed to update status. Please try again.')
  } finally {
    updatingStatusId.value = null
    updateStatusTarget.value = null
  }
}

// Debounced search
let searchTimeout: number
const debouncedSearch = () => {
  clearTimeout(searchTimeout)
  searchTimeout = setTimeout(async () => {
    if (searchQuery.value.trim()) {
      isSearching.value = true
      await searchAnomaliesQuery(searchQuery.value.trim())
    } else {
      isSearching.value = false
      loadAnomalies()
    }
  }, 300)
}

onMounted(() => {
  console.log('AnomalyTableView mounted, loading anomalies...')
  loadAnomalies()
})
</script>
