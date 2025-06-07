import { createApp } from 'vue'
import './style.css'
import App from './App.vue'
import router from './router'

console.log('Anomaly UI is running...')
createApp(App).use(router).mount('#app')
