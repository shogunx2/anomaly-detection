<template>
  <aside class="w-64 bg-slate-900 border-r border-slate-800 min-h-screen shadow-sm">
    <div class="p-6 border-b border-slate-800">
      <h2 class="text-xs font-bold uppercase tracking-widest text-slate-400">Navigation</h2>
    </div>
    
    <nav class="p-4 space-y-1">
      <ul class="space-y-1">
        <li v-for="item in navigationItems" :key="item.name">
          <!-- Use named-route navigation to avoid path/base mismatches -->
          <router-link
            :to="{ name: item.name }"
            class="nav-item"
            :class="[
              $route.name === item.name 
                ? 'bg-primary-600/10 text-primary-400 border-l-2 border-primary-600' 
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
            ]"
          >
            <component :is="item.icon" class="w-5 h-5 flex-shrink-0" />
            <span class="font-medium">{{ item.label }}</span>
          </router-link>
        </li>
      </ul>
    </nav>
  </aside>
</template>

<script setup lang="ts">
import { Grid3X3, List, BarChart3 } from 'lucide-vue-next'

const navigationItems = [
  {
    name: 'overview',
    label: 'Overview',
    to: '/',
    icon: Grid3X3
  },
  {
    name: 'distributions',
    label: 'Distributions',
    to: '/distributions',
    icon: BarChart3
  },
  {
    name: 'anomaly-table',
    label: 'Anomaly Table',
    to: '/anomaly-table',
    icon: List
  }
]
</script>
