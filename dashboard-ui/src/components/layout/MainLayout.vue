<template>
  <div class="flex h-screen bg-slate-950">
    <!-- Sidebar -->
    <Sidebar />
    
    <!-- Main Content -->
    <div class="flex-1 flex flex-col overflow-hidden">
      <!-- Header -->
      <Header />
      
      <!-- Page Content -->
      <main class="flex-1 overflow-auto bg-slate-950 px-6 py-4">
        <div class="max-w-5xl">
          <router-view />
        </div>
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import Header from './Header.vue'
import Sidebar from './Sidebar.vue'
</script>
