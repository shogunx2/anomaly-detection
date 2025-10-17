<template>
  <div class="space-y-4">
    <!-- Page Header -->
    <div class="section-header">
      <h1 class="section-title">Distributions</h1>
      <p class="section-subtitle">Anomaly status and severity distribution</p>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="flex justify-center items-center h-64">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="bg-danger-500/10 border border-danger-500/30 rounded-lg p-6">
      <p class="text-danger-400 font-medium">Error loading distributions</p>
      <p class="text-danger-300 text-sm mt-1">{{ error }}</p>
      <button 
        @click="refreshData"
        class="btn-primary mt-4"
      >
        Retry
      </button>
    </div>

    <!-- Main Content -->
    <div v-else>
      <!-- Distribution Charts -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <!-- Status Distribution -->
        <DonutChart
          :data="statusDistributionData"
          title="Status Distribution"
          subtitle="Distribution of anomaly statuses"
          :colors="statusColors"
        />

        <!-- Severity Distribution -->
        <DonutChart
          :data="severityDistributionData"
          title="Severity Distribution"
          subtitle="Distribution of anomaly severities"
          :colors="severityColors"
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

const statusDistributionData = computed(() => {
  if (!distributions.value?.status_distribution) return []
  
  return Object.entries(distributions.value.status_distribution).map(([label, value]) => ({
    label,
    value: value as number
  }))
})

const severityDistributionData = computed(() => {
  if (!overview.value?.severity_distribution) return []
  
  return Object.entries(overview.value.severity_distribution).map(([label, value]) => ({
    label,
    value: value as number
  }))
})

const refreshData = async () => {
  await Promise.all([
    fetchDistributions(),
    fetchOverview()
  ])
}

onMounted(async () => {
  console.log('DistributionsView mounted, checking data...')
  // Data is already loaded by App.vue, but refresh if needed
  if (!distributions.value || !overview.value) {
    await Promise.all([
      fetchDistributions(),
      fetchOverview()
    ])
  }
})
</script>
