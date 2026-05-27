import { ref } from 'vue'

export function useSSE() {
  const isResearching = ref(false)
  const subQuestions = ref([])
  const steps = ref([])
  const crossCheck = ref(null)
  const report = ref(null)
  const error = ref(null)

  function startResearch(query) {
    isResearching.value = true
    subQuestions.value = []
    steps.value = []
    crossCheck.value = null
    report.value = null
    error.value = null

    fetch('/api/research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    }).then(async (response) => {
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        let currentEvent = ''
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim()
          } else if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              handleEvent(currentEvent, data)
            } catch (e) {
              // Skip unparseable lines
            }
          }
        }
      }

      isResearching.value = false
    }).catch((err) => {
      error.value = err.message || 'Connection failed'
      isResearching.value = false
    })
  }

  function handleEvent(event, data) {
    switch (event) {
      case 'plan':
        subQuestions.value = data.sub_questions || []
        break
      case 'step':
        steps.value.push(data)
        break
      case 'cross_check':
        crossCheck.value = data
        break
      case 'report':
        report.value = data
        break
      case 'error':
        error.value = data.message || 'Unknown error'
        isResearching.value = false
        break
    }
  }

  return {
    isResearching,
    subQuestions,
    steps,
    crossCheck,
    report,
    error,
    startResearch,
  }
}
