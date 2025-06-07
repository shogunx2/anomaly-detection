<template>
  <div>
    <h2>Query Anomalies by Enterprise ID</h2>
    <input v-model="enterpriseId" placeholder="Enter enterprise_id" />
    <button @click="fetchAnomalies">Search</button>
    <table v-if="anomalies.length">
      <thead>
        <tr>
          <th>ID</th>
          <th>Resource Name</th>
          <th>Severity</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="anomaly in anomalies" :key="anomaly.id" @click="goToDetail(anomaly.id)" style="cursor:pointer">
          <td>{{ anomaly.id }}</td>
          <td>{{ anomaly.resource_name }}</td>
          <td>{{ anomaly.severity }}</td>
          <td>{{ anomaly.status }}</td>
        </tr>
      </tbody>
    </table>
    <div v-if="error" style="color:red">{{ error }}</div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'
import { useRouter } from 'vue-router'

const enterpriseId = ref('')
const anomalies = ref([])
const error = ref('')
const router = useRouter()

const fetchAnomalies = async () => {
  error.value = ''
  anomalies.value = []
  if (!enterpriseId.value) {
    error.value = 'Please enter an enterprise_id'
    return
  }
  try {
    const res = await axios.get(`http://localhost:8000/anomalies/${enterpriseId.value}`)
    anomalies.value = res.data
    if (!anomalies.value.length) error.value = 'No anomalies found.'
  } catch (e) {
    error.value = 'Error fetching anomalies'
  }
}

const goToDetail = (id) => {
  router.push({ name: 'AnomalyDetail', params: { id } })
}
</script>