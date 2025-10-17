<template>
  <transition
    enter-active-class="transition ease-out duration-300"
    enter-from-class="transform opacity-0 translate-y-2"
    enter-to-class="transform opacity-100 translate-y-0"
    leave-active-class="transition ease-in duration-200"
    leave-from-class="transform opacity-100 translate-y-0"
    leave-to-class="transform opacity-0 translate-y-2"
  >
    <div
      v-if="visible && currentNotification"
      class="fixed top-20 right-6 bg-red-600 text-white px-6 py-4 rounded-lg shadow-lg max-w-sm z-50"
    >
      <div class="flex items-start justify-between gap-4">
        <div class="flex-1">
          <div class="font-semibold text-lg">High-Severity Anomaly Detected</div>
          <div class="text-sm text-red-100 mt-1">
            {{ currentNotification.message }}
          </div>
          <div class="text-xs text-red-200 mt-2">
            Resource: <span class="font-mono">{{ currentNotification.resourceName }}</span>
          </div>
        </div>
        <button
          @click="hideNotification"
          class="text-red-200 hover:text-white transition-colors flex-shrink-0"
        >
          <X class="w-5 h-5" />
        </button>
      </div>
    </div>
  </transition>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { X } from 'lucide-vue-next'
import { storeToRefs } from 'pinia'
import { useDashboardStore } from '@/stores/dashboard'

const dashboardStore = useDashboardStore()
const { notifications } = storeToRefs(dashboardStore)

const visible = ref(false)
const currentNotificationIndex = ref(-1)
const hideTimeout = ref<ReturnType<typeof setTimeout> | null>(null)

const currentNotification = computed(() => {
  if (currentNotificationIndex.value >= 0 && currentNotificationIndex.value < notifications.value.length) {
    return notifications.value[currentNotificationIndex.value]
  }
  return null
})

const showNextNotification = () => {
  if (currentNotificationIndex.value + 1 < notifications.value.length) {
    // Show next notification
    currentNotificationIndex.value += 1
    visible.value = true
    scheduleHide()
  } else if (notifications.value.length > 0) {
    // Show first notification if none shown yet
    currentNotificationIndex.value = 0
    visible.value = true
    scheduleHide()
  }
}

const hideNotification = () => {
  visible.value = false
  if (hideTimeout.value) {
    clearTimeout(hideTimeout.value)
  }
}

const scheduleHide = () => {
  if (hideTimeout.value) {
    clearTimeout(hideTimeout.value)
  }
  // Auto-hide after 6 seconds
  hideTimeout.value = setTimeout(() => {
    visible.value = false
    // Show next notification if available
    setTimeout(() => {
      showNextNotification()
    }, 300)
  }, 6000)
}

// Watch for new notifications
watch(
  () => notifications.value.length,
  (newLength) => {
    if (newLength > 0) {
      console.debug('[NotificationBanner] New notifications detected, count:', newLength)
      if (!visible.value) {
        showNextNotification()
      }
    }
  }
)
</script>
