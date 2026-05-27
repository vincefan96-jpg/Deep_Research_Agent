<template>
  <div class="report-view" v-if="report">
    <h3>调研报告</h3>
    <div class="report-body" v-html="rendered"></div>
    <div class="sources" v-if="report.sources?.length">
      <h4>参考来源</h4>
      <ul>
        <li v-for="(s, i) in report.sources" :key="i">
          <a :href="s" target="_blank">{{ s }}</a>
        </li>
      </ul>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { marked } from 'marked'

const props = defineProps({
  report: { type: Object, default: null },
})

const rendered = computed(() => {
  if (!props.report?.markdown) return ''
  return marked(props.report.markdown)
})
</script>

<style scoped>
.report-view { padding: 1rem 0; }
.report-body { line-height: 1.7; }
.report-body :deep(h1) { font-size: 1.5rem; margin-top: 1rem; }
.report-body :deep(h2) { font-size: 1.25rem; margin-top: 1rem; }
.report-body :deep(h3) { font-size: 1.1rem; margin-top: 0.75rem; }
.report-body :deep(p) { margin: 0.5rem 0; }
.report-body :deep(code) { background: #f1f5f9; padding: 2px 6px; border-radius: 3px; font-size: 0.85rem; }
.sources { margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #e5e7eb; }
.sources ul { list-style: none; padding: 0; }
.sources li { margin: 0.25rem 0; font-size: 0.8rem; word-break: break-all; }
</style>
