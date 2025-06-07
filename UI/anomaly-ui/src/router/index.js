import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import AnomalyDetail from '../views/AnomalyDetail.vue'

const routes = [
  { path: '/', name: 'Home', component: Home },
  { path: '/anomaly/:id', name: 'AnomalyDetail', component: AnomalyDetail, props: true }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router