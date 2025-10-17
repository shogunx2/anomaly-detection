<template>
  <div class="space-y-4">
    <!-- Page Header -->
    <div class="section-header">
      <h1 class="section-title">Overview Dashboard</h1>
      <p class="section-subtitle">Key metrics and anomaly trends</p>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="flex justify-center items-center h-64">
      <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="bg-danger-500/10 border border-danger-500/30 rounded-lg p-6">
      <p class="text-danger-400 font-medium">Error loading dashboard</p>
      <p class="text-danger-300 text-sm mt-1">{{ error }}</p>
      <button 
        @click="refreshData"
        class="btn-primary mt-4"
      >
        Retry
      </button>
    </div>

    <!-- Main Content -->
    <div v-else class="space-y-4">
      <!-- Key Metrics -->
      <div class="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <MetricCard
          :label="'Total Anomalies'"
          :value="totalAnomalies"
          :description="'All time'"
          :icon="Info"
          :icon-color="'text-slate-300'"
        />
        
        <MetricCard
          :label="'Open'"
          :value="openAnomalies"
          :description="'Status: Open'"
          :icon="AlertTriangle"
          :icon-color="'text-slate-300'"
          :border-color="'bg-danger-500'"
        />
        
        <MetricCard
          :label="'Acknowledged'"
          :value="acknowledgedAnomalies"
          :description="'Status: Acknowledged'"
          :icon="Clock"
          :icon-color="'text-slate-300'"
          :border-color="'bg-warning-500'"
        />
        
        <MetricCard
          :label="'Resolved'"
          :value="resolvedAnomalies"
          :description="'Status: Resolved'"
          :icon="CheckCircle"
          :icon-color="'text-slate-300'"
          :border-color="'bg-success-500'"
        />
      </div>

      <!-- Timeline Chart -->
      <div class="w-full">
        <TimelineChart
          :data="timeline"
          title="Anomaly Timeline"
          subtitle="Anomaly counts over the last 7 days"
        />
      </div>

      <!-- Recent Activity Summary -->
      <div class="card">
        <h3 class="text-base font-semibold text-slate-50 mb-3">Recent Activity</h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div class="p-4 bg-slate-800/50 rounded-lg border border-slate-700 hover:border-slate-600 transition-colors">
            <p class="text-2xl font-bold text-slate-50 tracking-tight">{{ recentAnomalies }}</p>
            <p class="text-xs text-slate-400 mt-1 font-medium">Last 24 hours</p>
          </div>
          <div class="p-4 bg-slate-800/50 rounded-lg border border-slate-700 hover:border-slate-600 transition-colors">
            <p class="text-2xl font-bold text-slate-50 tracking-tight">{{ totalAnomalies }}</p>
            <p class="text-xs text-slate-400 mt-1 font-medium">This week</p>
          </div>
          <div class="p-4 bg-slate-800/50 rounded-lg border border-slate-700 hover:border-slate-600 transition-colors">
            <p class="text-2xl font-bold text-slate-50 tracking-tight">{{ totalAnomalies }}</p>
            <p class="text-xs text-slate-400 mt-1 font-medium">This month</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, computed } from 'vue'
import { storeToRefs } from 'pinia'
import { useDashboardStore } from '@/stores/dashboard'
import MetricCard from '@/components/MetricCard.vue'
import TimelineChart from '@/components/charts/TimelineChart.vue'
import { 
  Info, 
  AlertTriangle, 
  Clock, 
  CheckCircle 
} from 'lucide-vue-next'

const dashboardStore = useDashboardStore()

// Functions can be taken directly from the store
const { fetchOverview, fetchTimeline, refreshAll } = dashboardStore

// Use storeToRefs to preserve reactivity for state/computed values
const { loading, error, totalAnomalies, openAnomalies, acknowledgedAnomalies, resolvedAnomalies, recentAnomalies, timeline } = storeToRefs(dashboardStore)

const refreshData = async () => {
  await refreshAll()
}

onMounted(async () => {
  console.log('OverviewView mounted, checking data...')
  // Data is already loaded by App.vue, but refresh if needed
  if (!timeline.value.length) {
    await Promise.all([
      fetchOverview(),
      fetchTimeline()
    ])
  }
})
</script>
