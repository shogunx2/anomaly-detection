<template>
  <div class="metric-card">
    <div class="flex items-center justify-between">
      <div class="flex-1">
        <p class="metric-label">
          {{ label }}
        </p>
        <p class="metric-value text-2xl">
          {{ value }}
        </p>
        <p v-if="description" class="text-xs text-slate-500 mt-1">
          {{ description }}
        </p>
      </div>
      <div v-if="icon" class="flex-shrink-0 ml-3">
        <component 
          :is="icon" 
          class="w-6 h-6"
          :class="iconColor || 'text-primary-400'"
        />
      </div>
    </div>
    <div v-if="borderColor" class="absolute left-0 top-0 bottom-0 w-1 rounded-l-lg" :class="borderColor"></div>
  </div>
</template>

<script setup lang="ts">
import type { Component } from 'vue'

interface Props {
  label: string
  value: number | string
  description?: string
  icon?: Component
  iconColor?: string
  borderColor?: string
}

defineProps<Props>()
</script>

<style scoped>
.metric-card {
  @apply relative;
}
</style>
