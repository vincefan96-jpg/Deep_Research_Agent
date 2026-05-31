<template>
  <div class="app-shell">
    <header class="topbar">
      <span class="logo">深度调研智能体</span>
      <nav class="tabs">
        <button :class="{ active: view === 'chat' }" @click="view = 'chat'">调研</button>
        <button :class="{ active: view === 'history' }" @click="goHistoryList">历史</button>
      </nav>
    </header>

    <div class="app-layout" v-if="view === 'chat'">
      <aside class="sidebar">
        <StepTimeline :steps="steps" :isResearching="isResearching" />
      </aside>
      <main class="main-content">
        <ChatPanel
          :isResearching="isResearching"
          :steps="steps"
          :subQuestions="subQuestions"
          :report="report"
          :error="error"
          @ask="handleAsk"
        />
      </main>
    </div>

    <div class="app-layout" v-else-if="view === 'history'">
      <aside class="sidebar">
        <HistoryList
          :sessions="sessions"
          :selectedId="selectedSessionId"
          @select="selectSession"
          @delete="deleteSession"
        />
      </aside>
      <main class="main-content">
        <HistoryDetail
          :sessionId="selectedSessionId"
          @back="goHistoryList"
        />
      </main>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import ChatPanel from './components/ChatPanel.vue'
import StepTimeline from './components/StepTimeline.vue'
import HistoryList from './components/HistoryList.vue'
import HistoryDetail from './components/HistoryDetail.vue'
import { useSSE } from './composables/useSSE.js'

const view = ref('chat')
const sessions = ref([])
const selectedSessionId = ref(null)

const {
  isResearching,
  subQuestions,
  steps,
  report,
  error,
  startResearch,
} = useSSE()

function handleAsk(query) {
  view.value = 'chat'
  startResearch(query)
}

async function goHistoryList() {
  view.value = 'history'
  selectedSessionId.value = null
  const res = await fetch('/api/history')
  const data = await res.json()
  sessions.value = data.sessions || []
}

function selectSession(id) {
  selectedSessionId.value = id
}

async function deleteSession(id) {
  await fetch(`/api/history/${id}`, { method: 'DELETE' })
  if (selectedSessionId.value === id) {
    selectedSessionId.value = null
  }
  sessions.value = sessions.value.filter(s => s.id !== id)
}
</script>

<style scoped>
.app-shell { display: flex; flex-direction: column; height: 100vh; }
.topbar { display: flex; align-items: center; gap: 1.5rem; padding: 0 1.5rem; height: 48px; border-bottom: 1px solid #e5e7eb; background: white; flex-shrink: 0; }
.logo { font-weight: 700; font-size: 1rem; color: #0ea5e9; }
.tabs { display: flex; gap: 0.25rem; }
.tabs button { padding: 6px 16px; border: none; background: none; cursor: pointer; font-size: 0.85rem; border-radius: 6px; color: #6b7280; }
.tabs button:hover { background: #f3f4f6; }
.tabs button.active { background: #f0f9ff; color: #0ea5e9; font-weight: 600; }
.app-layout { display: flex; flex: 1; min-height: 0; }
.sidebar { width: 380px; border-right: 1px solid #e5e7eb; overflow-y: auto; background: #fafafa; flex-shrink: 0; }
.main-content { flex: 1; overflow-y: auto; }
@media (max-width: 768px) {
  .app-layout { flex-direction: column; }
  .sidebar { width: 100%; max-height: 40vh; border-right: none; border-bottom: 1px solid #e5e7eb; }
}
</style>
