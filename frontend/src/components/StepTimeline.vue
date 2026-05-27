<template>
  <div class="step-timeline">
    <h3 v-if="steps.length">调研步骤</h3>
    <div v-for="(step, i) in steps" :key="i" :class="['step-card', step.type]">
      <div class="step-header">
        <span class="step-badge">{{ typeLabel(step.type) }}</span>
        <span class="step-round">第 {{ step.round }} 轮</span>
      </div>
      <div class="step-content">
        <template v-if="step.type === 'action' && step.tool_name">
          <strong>{{ step.tool_name }}</strong>
          <pre v-if="step.tool_params">{{ JSON.stringify(step.tool_params, null, 2) }}</pre>
        </template>
        <template v-else>
          {{ step.content?.slice(0, 300) }}{{ step.content?.length > 300 ? '...' : '' }}
        </template>
      </div>
    </div>
    <div v-if="!steps.length && isResearching" class="loading">正在启动调研...</div>
  </div>
</template>

<script setup>
defineProps({
  steps: { type: Array, default: () => [] },
  isResearching: { type: Boolean, default: false },
})

function typeLabel(type) {
  const labels = { thought: '思考', action: '行动', observation: '观察' }
  return labels[type] || type.toUpperCase()
}
</script>

<style scoped>
.step-timeline { padding: 0.5rem 0; }
.step-card {
  border-left: 3px solid #ddd;
  padding: 0.5rem 1rem;
  margin-bottom: 0.5rem;
  border-radius: 0 6px 6px 0;
}
.step-card.thought { border-color: #6366f1; background: #f5f3ff; }
.step-card.action { border-color: #f59e0b; background: #fffbeb; }
.step-card.observation { border-color: #10b981; background: #f0fdf4; }
.step-header { display: flex; gap: 0.5rem; align-items: center; margin-bottom: 0.25rem; }
.step-badge { font-size: 0.7rem; font-weight: 700; padding: 2px 6px; border-radius: 4px; }
.thought .step-badge { background: #e0e7ff; color: #4338ca; }
.action .step-badge { background: #fef3c7; color: #b45309; }
.observation .step-badge { background: #d1fae5; color: #065f46; }
.step-round { font-size: 0.75rem; color: #888; }
.step-content { font-size: 0.85rem; line-height: 1.5; white-space: pre-wrap; }
pre { background: #1e1e1e; color: #d4d4d4; padding: 0.5rem; border-radius: 4px; font-size: 0.75rem; overflow-x: auto; }
.loading { color: #888; font-style: italic; padding: 1rem; }
</style>
