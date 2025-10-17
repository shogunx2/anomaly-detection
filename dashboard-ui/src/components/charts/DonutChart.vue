<template>
  <div class="card">
    <div class="mb-3">
      <h3 class="text-base font-semibold text-slate-50">{{ title }}</h3>
      <p class="text-xs text-slate-400 mt-0.5">{{ subtitle }}</p>
    </div>
    
    <div class="h-56 relative">
      <canvas ref="chartRef"></canvas>
    </div>
      <div v-if="!props.data || props.data.length === 0" class="absolute inset-0 flex items-center justify-center text-slate-400">
        No distribution data
      </div>
    
    <!-- Legend -->
    <div v-if="showLegend" class="mt-3 flex flex-wrap gap-3 justify-center">
      <div 
        v-for="(item, index) in legendItems" 
        :key="item.label"
        class="flex items-center space-x-1.5"
      >
        <div 
          class="w-2 h-2 rounded-full"
          :style="{ backgroundColor: item.color }"
        ></div>
        <span class="text-xs text-slate-400">{{ item.label }} ({{ item.value }})</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, computed } from 'vue'
import {
  Chart,
  ArcElement,
  Tooltip,
  Legend,
  DoughnutController
} from 'chart.js'

Chart.register(ArcElement, Tooltip, Legend, DoughnutController)
console.debug('[DonutChart] Chart.js registered with ArcElement, Tooltip, Legend, DoughnutController')

interface ChartData {
  label: string
  value: number
}

interface Props {
  data: ChartData[]
  title?: string
  subtitle?: string
  showLegend?: boolean
  colors?: string[]
}

const props = withDefaults(defineProps<Props>(), {
  title: 'Donut Chart',
  subtitle: 'Data distribution',
  showLegend: true,
  colors: () => ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899']
})

const chartRef = ref<HTMLCanvasElement>()
let chart: Chart | null = null

const legendItems = computed(() => {
  return props.data.map((item, index) => ({
    label: item.label,
    value: item.value,
    color: props.colors[index % props.colors.length]
  }))
})

const createChart = () => {
  console.debug('[DonutChart.createChart] Starting chart creation')
  if (!chartRef.value) {
    console.warn('[DonutChart.createChart] chartRef.value is null, returning early')
    return
  }

  const ctx = chartRef.value.getContext('2d')
  if (!ctx) {
    console.error('[DonutChart.createChart] Failed to get 2D context from canvas')
    return
  }

  console.debug('[DonutChart.createChart] Canvas context obtained, creating chart with data:', {
    dataLength: props.data?.length || 0,
    labels: props.data?.map(d => d.label) || [],
    values: props.data?.map(d => d.value) || [],
    hasData: !!(props.data && props.data.length)
  })

  try {
    chart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: props.data.map(d => d.label),
        // Ensure at least one dataset exists so `chart.data.datasets[0]` is safe to access
        datasets: (props.data && props.data.length) ? [
          {
            data: props.data.map(d => d.value),
            backgroundColor: props.data.map((_, index) => props.colors[index % props.colors.length]),
            borderColor: '#1f2937',
            borderWidth: 2
          }
        ] : [
          {
            data: [],
            backgroundColor: [],
            borderColor: '#1f2937',
            borderWidth: 2
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false // We're using custom legend
          },
          tooltip: {
            backgroundColor: '#374151',
            titleColor: '#f9fafb',
            bodyColor: '#f9fafb',
            borderColor: '#6b7280',
            borderWidth: 1
          }
        },
        cutout: '60%'
      }
    })
    console.debug('[DonutChart.createChart] Chart instance created successfully')
  } catch (error) {
    console.error('[DonutChart.createChart] Error creating chart:', error)
    throw error
  }
}

const updateChart = () => {
  console.debug('[DonutChart.updateChart] Called with data:', {
    dataLength: props.data?.length || 0,
    hasChart: !!chart
  })

  if (!chart) {
    console.warn('[DonutChart.updateChart] Chart instance is null, returning')
    return
  }

  const ds = chart.data.datasets as any[]
  if (!props.data || !props.data.length) {
    console.debug('[DonutChart.updateChart] No data provided, clearing chart')
    chart.data.labels = []
    chart.data.datasets = []
    chart.update()
    return
  }

  console.debug('[DonutChart.updateChart] Updating chart with new data:', {
    dataLength: props.data.length,
    labels: props.data.map(d => d.label),
    values: props.data.map(d => d.value)
  })

  try {
    chart.data.labels = props.data.map(d => d.label)

    const newData = props.data.map(d => d.value)
    const newBg = props.data.map((_, index) => props.colors[index % props.colors.length])

    if (!ds || !ds[0]) {
      console.debug('[DonutChart.updateChart] Creating new dataset (no existing dataset found)')
      chart.data.datasets = [
        {
          data: newData,
          backgroundColor: newBg,
          borderColor: '#1f2937',
          borderWidth: 2
        }
      ]
    } else {
      console.debug('[DonutChart.updateChart] Updating existing dataset')
      ds[0].data = newData
      ds[0].backgroundColor = newBg
    }

    chart.update()
    console.debug('[DonutChart.updateChart] Chart updated successfully')
  } catch (error) {
    console.error('[DonutChart.updateChart] Error updating chart:', error)
  }
}

onMounted(() => {
  console.debug('[DonutChart.onMounted] Component mounted, initializing chart')
  try {
    createChart()
  } catch (error) {
    console.error('[DonutChart.onMounted] Error during chart creation:', error)
  }
})

watch(() => props.data, updateChart, { deep: true })
</script>
