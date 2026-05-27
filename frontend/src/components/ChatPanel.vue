<template>
  <div class="chat-panel">
    <div class="messages">
      <div class="empty-state" v-if="!report && !isResearching">
        <h2>Deep Research Agent</h2>
        <p>Ask any research question and I'll investigate it thoroughly.</p>
        <div class="examples">
          <button v-for="q in exampleQuestions" :key="q" @click="$emit('ask', q)" class="example-btn">
            {{ q }}
          </button>
        </div>
      </div>

      <div v-if="subQuestions.length" class="plan-card">
        <h4>Research Plan</h4>
        <ol>
          <li v-for="(q, i) in subQuestions" :key="i">{{ q }}</li>
        </ol>
      </div>

      <div v-if="crossCheck" class="cross-check-card">
        <h4>Cross-Check</h4>
        <p>Consistency: <strong>{{ crossCheck.consistency }}</strong></p>
        <ul v-if="crossCheck.conflicts?.length">
          <li v-for="(c, i) in crossCheck.conflicts" :key="i">⚠ {{ c }}</li>
        </ul>
      </div>

      <ReportView :report="report" />

      <div v-if="error" class="error-card">
        <strong>Error:</strong> {{ error }}
      </div>

      <div v-if="isResearching && !report" class="researching-indicator">
        <span class="pulse"></span> Researching... ({{ steps.length }} steps)
      </div>
    </div>

    <form @submit.prevent="handleSubmit" class="input-area">
      <input
        v-model="query"
        type="text"
        placeholder="Enter your research question..."
        :disabled="isResearching"
      />
      <button type="submit" :disabled="isResearching || !query.trim()">Research</button>
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
  crossCheck: { type: Object, default: null },
  report: { type: Object, default: null },
  error: { type: String, default: null },
})

const emit = defineEmits(['ask'])

const query = ref('')

const exampleQuestions = [
  'What are the main trends in AI agent development in 2025?',
  'Compare React, Vue, and Svelte for building large-scale applications',
  'What is the current state of quantum computing research?',
]

function handleSubmit() {
  if (query.value.trim()) {
    emit('ask', query.value.trim())
    query.value = ''
  }
}
</script>

<style scoped>
.chat-panel { display: flex; flex-direction: column; height: 100vh; }
.messages { flex: 1; overflow-y: auto; padding: 1.5rem; }
.empty-state { text-align: center; padding: 3rem 1rem; }
.empty-state h2 { font-size: 1.5rem; margin-bottom: 0.5rem; }
.empty-state p { color: #666; margin-bottom: 1.5rem; }
.examples { display: flex; flex-direction: column; gap: 0.5rem; align-items: center; }
.example-btn { background: #f1f5f9; border: 1px solid #e2e8f0; padding: 0.5rem 1rem; border-radius: 8px; cursor: pointer; font-size: 0.85rem; max-width: 500px; text-align: left; }
.example-btn:hover { background: #e2e8f0; }
.plan-card, .cross-check-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
.error-card { background: #fef2f2; border: 1px solid #fecaca; color: #dc2626; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
.researching-indicator { display: flex; align-items: center; gap: 0.5rem; color: #6366f1; padding: 1rem; }
.pulse { width: 10px; height: 10px; background: #6366f1; border-radius: 50%; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.input-area { display: flex; gap: 0.5rem; padding: 1rem; border-top: 1px solid #e5e7eb; }
.input-area input { flex: 1; padding: 0.75rem; border: 1px solid #d1d5db; border-radius: 8px; font-size: 0.95rem; }
.input-area button { padding: 0.75rem 1.5rem; background: #6366f1; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; }
.input-area button:disabled { background: #a5b4fc; cursor: not-allowed; }
</style>
