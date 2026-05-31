<template>
  <div class="history-list">
    <h2>历史记录</h2>
    <div class="list">
      <div
        v-for="s in sessions"
        :key="s.id"
        class="history-item"
        :class="{ active: selectedId === s.id }"
      >
        <div class="item-main" @click="$emit('select', s.id)">
          <div class="query-text">{{ s.query }}</div>
          <div class="meta">
            <span class="status" :class="s.status">{{ statusLabel(s.status) }}</span>
            <span>{{ s.created_at }}</span>
          </div>
        </div>
        <button class="delete-btn" @click.stop="$emit('delete', s.id)" title="删除记录">×</button>
      </div>
      <div v-if="sessions.length === 0" class="empty">暂无历史记录</div>
    </div>
  </div>
</template>

<script setup>
defineProps({
  sessions: { type: Array, default: () => [] },
  selectedId: { type: String, default: null },
})

defineEmits(['select', 'delete'])

function statusLabel(status) {
  const map = { running: '进行中', completed: '已完成', error: '出错' }
  return map[status] || status
}
</script>

<style scoped>
.history-list { padding: 1rem; }
.history-list h2 { font-size: 1.1rem; margin-bottom: 0.75rem; color: #374151; }
.list { display: flex; flex-direction: column; gap: 0.5rem; }
.history-item {
  display: flex; align-items: center;
  border-radius: 8px; border: 1px solid #e5e7eb; background: white;
  transition: border-color 0.15s; overflow: hidden;
}
.history-item:hover { border-color: #0ea5e9; }
.history-item.active { border-color: #0ea5e9; background: #f0f9ff; }
.item-main { flex: 1; padding: 0.75rem; cursor: pointer; min-width: 0; }
.query-text { font-size: 0.9rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.meta { display: flex; gap: 0.75rem; margin-top: 0.35rem; font-size: 0.75rem; color: #9ca3af; }
.status.completed { color: #16a34a; }
.status.running { color: #0ea5e9; }
.status.error { color: #dc2626; }
.delete-btn {
  padding: 0.5rem 0.75rem; border: none; background: none;
  color: #d1d5db; font-size: 1.25rem; cursor: pointer;
  flex-shrink: 0; align-self: stretch;
  transition: color 0.15s, background 0.15s;
}
.delete-btn:hover { color: #ef4444; background: #fef2f2; }
.empty { color: #9ca3af; text-align: center; padding: 2rem 0; }
</style>
