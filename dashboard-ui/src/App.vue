<script setup lang="ts">
import { onMounted } from 'vue'
import MainLayout from './components/layout/MainLayout.vue'
import { useDashboardStore } from './stores/dashboard'

const dashboardStore = useDashboardStore()

onMounted(async () => {
  console.log('App mounted, initializing dashboard...')
  
  // Initialize the dashboard with data
  try {
    await dashboardStore.refreshAll()
    console.log('Dashboard initialized successfully')
  } catch (error) {
    console.error('Failed to initialize dashboard:', error)
  }
  
  // Set up periodic health checks every 30 seconds
  setInterval(() => {
    dashboardStore.checkHealth()
  }, 30000)
})
</script>

<template>
  <!-- Let MainLayout render the page content via its own <router-view /> -->
  <MainLayout />
</template>
