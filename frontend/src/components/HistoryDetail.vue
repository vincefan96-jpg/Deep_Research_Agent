<template>
  <div class="history-detail">
    <div class="back-bar">
      <button @click="$emit('back')" class="back-btn">← 返回列表</button>
    </div>
    <div v-if="loading" class="loading">加载中...</div>
    <template v-else-if="session">
      <h2>{{ session.query }}</h2>
      <div class="meta-bar">
        <span class="status" :class="session.status">{{ statusLabel(session.status) }}</span>
        <span>{{ session.created_at }}</span>
      </div>
      <StepTimeline :steps="session.steps || []" :isResearching="false" />
      <ReportView v-if="session.report" :report="{ markdown: session.report, sources: [] }" />
    </template>
    <div v-else class="empty">会话未找到</div>
  </div>
</template>

<script setup>
import { ref, watch } from 'vue'
import StepTimeline from './StepTimeline.vue'
import ReportView from './ReportView.vue'

const props = defineProps({
  sessionId: { type: String, default: null },
})

defineEmits(['back'])

const session = ref(null)
const loading = ref(false)

async function fetchSession(id) {
  if (!id) { session.value = null; return }
  loading.value = true
  try {
    const res = await fetch(`/api/history/${id}`)
    if (res.ok) session.value = await res.json()
    else session.value = null
  } catch {
    session.value = null
  } finally {
    loading.value = false
  }
}

watch(() => props.sessionId, fetchSession, { immediate: true })

function statusLabel(status) {
  const map = { running: '进行中', completed: '已完成', error: '出错' }
  return map[status] || status
}
</script>

<style scoped>
.history-detail { padding: 1.5rem; }
.back-bar { margin-bottom: 1rem; }
.back-btn { background: none; border: none; color: #0ea5e9; cursor: pointer; font-size: 0.9rem; padding: 0; }
.back-btn:hover { text-decoration: underline; }
.loading { color: #9ca3af; padding: 2rem; text-align: center; }
.empty { color: #9ca3af; text-align: center; padding: 2rem; }
.meta-bar { display: flex; gap: 0.75rem; margin-bottom: 1rem; font-size: 0.8rem; color: #9ca3af; }
.status.completed { color: #16a34a; }
.status.error { color: #dc2626; }
.status.running { color: #0ea5e9; }
</style>
