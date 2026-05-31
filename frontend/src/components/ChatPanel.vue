<template>
  <div class="chat-panel">
    <div class="messages">
      <div class="empty-state" v-if="!report && !isResearching">
        <h2>深度调研智能体</h2>
        <p>提出任何研究问题，我会为你进行深入调研。</p>
        <div class="examples">
          <button v-for="q in exampleQuestions" :key="q" @click="$emit('ask', q)" class="example-btn">
            {{ q }}
          </button>
        </div>
      </div>

      <div v-if="subQuestions.length" class="plan-card">
        <h4>调研计划</h4>
        <ol>
          <li v-for="(q, i) in subQuestions" :key="i">{{ q }}</li>
        </ol>
      </div>

      <ReportView :report="report" />

      <div v-if="error" class="error-card">
        <strong>错误：</strong>{{ error }}
      </div>

      <div v-if="isResearching && !report" class="researching-indicator">
        <span class="pulse"></span> 调研中...（已执行 {{ steps.length }} 步）
      </div>
    </div>

    <form @submit.prevent="handleSubmit" class="input-area">
      <input
        v-model="query"
        type="text"
        placeholder="输入你的研究问题..."
        :disabled="isResearching"
      />
      <button type="submit" :disabled="isResearching || !query.trim()">开始调研</button>
    </form>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import ReportView from './ReportView.vue'

const props = defineProps({
  isResearching: Boolean,
  steps: { type: Array, default: () => [] },
  subQuestions: { type: Array, default: () => [] },
  report: { type: Object, default: null },
  error: { type: String, default: null },
})

const emit = defineEmits(['ask'])

const query = ref('')

const exampleQuestions = [
  '2025 年 AI Agent 开发的主要趋势是什么？',
  '对比 React、Vue 和 Svelte 在大规模应用中的表现',
  '量子计算研究的现状如何？',
]

function handleSubmit() {
  if (query.value.trim()) {
    emit('ask', query.value.trim())
    query.value = ''
  }
}
</script>

<style scoped>
.chat-panel { display: flex; flex-direction: column; height: 100%; }
.messages { flex: 1; overflow-y: auto; padding: 1.5rem; }
.empty-state { text-align: center; padding: 3rem 1rem; }
.empty-state h2 { font-size: 1.5rem; margin-bottom: 0.5rem; }
.empty-state p { color: #666; margin-bottom: 1.5rem; }
.examples { display: flex; flex-direction: column; gap: 0.5rem; align-items: center; }
.example-btn { background: #f1f5f9; border: 1px solid #e2e8f0; padding: 0.5rem 1rem; border-radius: 8px; cursor: pointer; font-size: 0.85rem; max-width: 500px; text-align: left; }
.example-btn:hover { background: #e2e8f0; }
.plan-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
.error-card { background: #fef2f2; border: 1px solid #fecaca; color: #dc2626; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
.researching-indicator { display: flex; align-items: center; gap: 0.5rem; color: #0ea5e9; padding: 1rem; }
.pulse { width: 10px; height: 10px; background: #0ea5e9; border-radius: 50%; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.input-area { display: flex; gap: 0.5rem; padding: 1rem; border-top: 1px solid #e5e7eb; }
.input-area input { flex: 1; padding: 0.75rem; border: 1px solid #d1d5db; border-radius: 8px; font-size: 0.95rem; }
.input-area button { padding: 0.75rem 1.5rem; background: #0ea5e9; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; }
.input-area button:disabled { background: #7dd3fc; cursor: not-allowed; }
</style>
