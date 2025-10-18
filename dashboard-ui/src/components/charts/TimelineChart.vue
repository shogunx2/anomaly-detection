<template>
  <div class="card">
    <div class="mb-3">
      <h3 class="text-base font-semibold text-slate-50">{{ title }}</h3>
      <p class="text-xs text-slate-400 mt-0.5">{{ subtitle }}</p>
    </div>
    
    <div class="h-56 relative">
        <canvas ref="chartRef"></canvas>
        <div v-if="!props.data || props.data.length === 0" class="absolute inset-0 flex items-center justify-center text-slate-400 pointer-events-none">
          No timeline data
        </div>
      </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import {
  Chart,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  LineController
} from 'chart.js'
import type { TimelineData } from '@/services/api'

Chart.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  LineController
)
console.debug('[TimelineChart] Chart.js registered with all required controllers and plugins')

interface Props {
  data: TimelineData[]
  title?: string
  subtitle?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Timeline Chart',
  subtitle: 'Data over time'
})

const chartRef = ref<HTMLCanvasElement>()
let chart: Chart | null = null

const createChart = () => {
  console.debug('[TimelineChart.createChart] Starting chart creation')
  if (!chartRef.value) {
    console.warn('[TimelineChart.createChart] chartRef.value is null, returning early')
    return
  }

  const ctx = chartRef.value.getContext('2d')
  if (!ctx) {
    console.error('[TimelineChart.createChart] Failed to get 2D context from canvas')
    return
  }

  console.debug('[TimelineChart.createChart] Canvas context obtained, creating chart with data:', {
    dataLength: props.data?.length || 0,
    dateLabels: props.data?.map(d => d.date) || [],
    counts: props.data?.map(d => d.count) || []
  })

  try {
    // If no data, create an empty chart with no datasets
    chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: (props.data || []).map(d => new Date(d.date).toLocaleDateString()),
        datasets: props.data && props.data.length ? [
          {
            label: 'Anomalies',
            data: (props.data || []).map(d => d.count),
            borderColor: '#0ea5e9',
            backgroundColor: 'rgba(14, 165, 233, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4
          }
        ] : []
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: {
              color: '#d1d5db'
            }
          }
        },
        scales: {
          x: {
            ticks: {
              color: '#9ca3af'
            },
            grid: {
              color: '#374151'
            }
          },
          y: {
            beginAtZero: true,
            ticks: {
              color: '#9ca3af'
            },
            grid: {
              color: '#374151'
            }
          }
        }
      }
    })
    console.debug('[TimelineChart.createChart] Chart instance created successfully')
  } catch (error) {
    console.error('[TimelineChart.createChart] Error creating chart:', error)
    throw error
  }
}

const updateChart = () => {
  console.debug('[TimelineChart.updateChart] Called with data:', {
    dataLength: props.data?.length || 0,
    hasChart: !!chart
  })

  if (!chart) {
    console.warn('[TimelineChart.updateChart] Chart instance is null')
    return
  }

  if (!chart.data.datasets[0]) {
    console.warn('[TimelineChart.updateChart] No dataset found in chart')
    return
  }

  console.debug('[TimelineChart.updateChart] Updating chart with new data:', {
    dataLength: props.data?.length || 0,
    newLabels: (props.data || []).map(d => new Date(d.date).toLocaleDateString()),
    newCounts: (props.data || []).map(d => d.count)
  })

  try {
    chart.data.labels = props.data.map(d => new Date(d.date).toLocaleDateString())
    chart.data.datasets[0].data = props.data.map(d => d.count)
    chart.update()
    console.debug('[TimelineChart.updateChart] Chart updated successfully')
  } catch (error) {
    console.error('[TimelineChart.updateChart] Error updating chart:', error)
  }
}

onMounted(() => {
  console.debug('[TimelineChart.onMounted] Component mounted, initializing chart')
  try {
    createChart()
  } catch (error) {
    console.error('[TimelineChart.onMounted] Error during chart creation:', error)
  }
})

watch(() => props.data, updateChart, { deep: true })
</script>
