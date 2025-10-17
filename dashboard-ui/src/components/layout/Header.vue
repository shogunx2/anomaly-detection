<template>
  <header class="bg-slate-900 border-b border-slate-800 px-6 py-4 shadow-sm">
    <div class="flex items-center justify-between">
      <!-- Left side - Title -->
      <div class="flex items-center space-x-4">
        <h1 class="text-2xl font-bold text-slate-50 tracking-tight">
          Anomaly Detection Dashboard
        </h1>
      </div>

      <!-- Right side - Connection Status -->
      <div class="flex items-center space-x-2">
        <div 
          class="w-2.5 h-2.5 rounded-full transition-colors shadow-lg"
          :class="isConnected ? 'bg-success-500 animate-pulse shadow-success-500/50' : 'bg-danger-500 shadow-danger-500/50'"
        ></div>
        <span class="text-sm font-medium text-slate-400">
          {{ isConnected ? 'Connected' : 'Disconnected' }}
        </span>
      </div>
    </div>
  </header>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { useDashboardStore } from '@/stores/dashboard'

const dashboardStore = useDashboardStore()
const { isConnected } = storeToRefs(dashboardStore)
</script>
