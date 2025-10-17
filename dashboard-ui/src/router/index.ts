import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'overview',
    component: () => import('../views/OverviewView.vue'),
  },
  {
    path: '/distributions',
    name: 'distributions',
    component: () => import('../views/DistributionsView.vue'),
  },
  {
    path: '/anomaly-table',
    name: 'anomaly-table',
    component: () => import('../views/AnomalyTableView.vue'),
  },
  // Fallback to overview for unknown routes
  {
    path: '/:pathMatch(.*)*',
    redirect: { name: 'overview' },
  },
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
  scrollBehavior() {
    return { top: 0 }
  },
  linkActiveClass: 'router-link-active',
  linkExactActiveClass: 'router-link-exact-active',
})

export default router