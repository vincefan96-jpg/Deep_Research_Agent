<template>
  <div class="app-layout">
    <aside class="sidebar">
      <StepTimeline :steps="steps" :isResearching="isResearching" />
    </aside>
    <main class="main-content">
      <ChatPanel
        :isResearching="isResearching"
        :steps="steps"
        :subQuestions="subQuestions"
        :crossCheck="crossCheck"
        :report="report"
        :error="error"
        @ask="handleAsk"
      />
    </main>
  </div>
</template>

<script setup>
import ChatPanel from './components/ChatPanel.vue'
import StepTimeline from './components/StepTimeline.vue'
import { useSSE } from './composables/useSSE.js'

const {
  isResearching,
  subQuestions,
  steps,
  crossCheck,
  report,
  error,
  startResearch,
} = useSSE()

function handleAsk(query) {
  startResearch(query)
}
</script>

<style scoped>
.app-layout { display: flex; height: 100vh; }
.sidebar { width: 380px; border-right: 1px solid #e5e7eb; overflow-y: auto; padding: 1rem; background: #fafafa; flex-shrink: 0; }
.main-content { flex: 1; overflow: hidden; }
@media (max-width: 768px) {
  .app-layout { flex-direction: column; }
  .sidebar { width: 100%; max-height: 40vh; border-right: none; border-bottom: 1px solid #e5e7eb; }
}
</style>
