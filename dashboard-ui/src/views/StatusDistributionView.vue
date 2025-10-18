<template>
  <div class="space-y-6">
    <!-- Page Header -->
    <div>
      <h1 class="text-2xl font-bold text-white">Status Distribution</h1>
      <p class="text-gray-400 mt-1">Anomaly status and severity distribution</p>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="flex justify-center items-center h-64">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="bg-red-900/50 border border-red-500 rounded-lg p-4">
      <p class="text-red-200">{{ error }}</p>
      <button 
        @click="refreshData"
        class="btn-primary mt-2"
      >
        Retry
      </button>
    </div>

    <!-- Main Content -->
    <div v-else>
      <!-- Distribution Charts -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Status Distribution -->
        <DonutChart
          :data="statusDistributionData"
          title="Status Distribution"
          subtitle="Distribution of anomaly statuses"
          :colors="statusColorsForChart"
        />

        <!-- Severity Distribution -->
        <DonutChart
          :data="severityDistributionData"
          title="Severity Distribution"
          subtitle="Distribution of anomaly severities"
          :colors="severityColorsForChart"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, computed } from 'vue'
import { storeToRefs } from 'pinia'
import { useDashboardStore } from '@/stores/dashboard'
import DonutChart from '@/components/charts/DonutChart.vue'

const dashboardStore = useDashboardStore()

const { fetchDistributions, fetchOverview } = dashboardStore
const { loading, error, overview, distributions } = storeToRefs(dashboardStore)

const statusColors = ['#ef4444', '#f97316', '#22c55e', '#3b82f6']
const severityColors = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899']

// Map status labels to their correct colors
const getStatusColor = (status: string): string => {
  const colorMap: { [key: string]: string } = {
    'Open': '#ef4444',      // red
    'Acknowledged': '#f97316', // orange
    'Resolved': '#22c55e'   // green
  }
  return colorMap[status] || '#9ca3af'
}

// Map severity labels to their correct colors
const getSeverityColor = (severity: string): string => {
  const colorMap: { [key: string]: string } = {
    'High': '#ef4444',      // red
    'Medium': '#f97316',    // orange
    'Low': '#22c55e'        // green
  }
  return colorMap[severity] || '#9ca3af'
}

const statusDistributionData = computed(() => {
  if (!distributions.value?.status_distribution) return []
  
  // Define consistent order for status
  const statusOrder = ['Open', 'Acknowledged', 'Resolved']
  const data = Object.entries(distributions.value.status_distribution).map(([label, value]) => ({
    label,
    value: value as number
  }))
  
  // Sort by the defined order
  return data.sort((a, b) => {
    const indexA = statusOrder.indexOf(a.label)
    const indexB = statusOrder.indexOf(b.label)
    return indexA - indexB
  })
})

const statusColorsForChart = computed(() => {
  // Map colors based on the actual label, not array position
  return statusDistributionData.value.map(item => {
    const colorMap: { [key: string]: string } = {
      'Open': '#ef4444',      // red
      'Acknowledged': '#f97316', // orange
      'Resolved': '#22c55e'   // green
    }
    return colorMap[item.label] || '#9ca3af'
  })
})

const severityDistributionData = computed(() => {
  if (!overview.value?.severity_distribution) return []
  
  // Define consistent order for severity
  const severityOrder = ['High', 'Medium', 'Low']
  const data = Object.entries(overview.value.severity_distribution).map(([label, value]) => ({
    label,
    value: value as number
  }))
  
  // Sort by the defined order
  return data.sort((a, b) => {
    const indexA = severityOrder.indexOf(a.label)
    const indexB = severityOrder.indexOf(b.label)
    return indexA - indexB
  })
})

const severityColorsForChart = computed(() => {
  // Map colors based on the actual label, not array position
  return severityDistributionData.value.map(item => {
    const colorMap: { [key: string]: string } = {
      'High': '#ef4444',      // red
      'Medium': '#f97316',    // orange
      'Low': '#22c55e'        // green
    }
    return colorMap[item.label] || '#9ca3af'
  })
})

const refreshData = async () => {
  await Promise.all([
    fetchDistributions(),
    fetchOverview()
  ])
}

onMounted(async () => {
  // Data is already loaded by App.vue, but refresh if needed
  if (!distributions.value || !overview.value) {
    await Promise.all([
      fetchDistributions(),
      fetchOverview()
    ])
  }
})
</script>
