<template>
  <div>
    <button @click="$router.back()">Back</button>
    <h2>Anomaly Details</h2>
    <div v-if="anomaly">
      <pre>{{ anomaly }}</pre>
    </div>
    <div v-else-if="error" style="color:red">{{ error }}</div>
    <div v-else>Loading...</div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { useRoute } from 'vue-router'

const route = useRoute()
const anomaly = ref(null)
const error = ref('')

onMounted(async () => {
  try {
    const res = await axios.get(`http://localhost:8000/anomaly/${route.params.id}`)
    anomaly.value = res.data
  } catch (e) {
    error.value = 'Error fetching anomaly details'
  }
})
</script>