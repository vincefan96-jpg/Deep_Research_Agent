# Deep Research Agent（深度调研智能体）

基于 ReAct 循环的 LLM 深度调研工具，自动进行多轮网络搜索、信息提取和综合分析，生成结构化的调研报告。

## 架构概览

```
用户输入问题 → 生成调研计划 → ReAct 循环（思考→搜索→阅读→分析）→ 生成报告
```

- **后端**：FastAPI + SSE 流式推送，实时展示调研过程
- **前端**：Vue 3 + Vite，支持实时步骤展示和历史记录管理
- **存储**：SQLite（记录调研历史）
- **LLM**：兼容 OpenAI API，支持自定义模型和 Base URL

## 项目结构

```
├── backend/
│   ├── agent/
│   │   ├── loop.py              # ReAct 循环引擎（Thought → Action → Observation）
│   │   ├── memory.py            # 上下文记忆管理
│   │   ├── orchestrator.py      # 调研编排器（计划 + 循环）
│   │   ├── parser.py            # LLM 输出解析器
│   │   └── tools/
│   │       ├── registry.py      # 工具注册表
│   │       ├── web_search.py    # 网络搜索工具
│   │       └── fetch_page.py    # 网页抓取工具
│   ├── db/
│   │   ├── connection.py        # SQLite 连接管理
│   │   └── repository.py        # 会话存储/查询
│   ├── llm/
│   │   └── client.py            # LLM 客户端（OpenAI 兼容）
│   ├── models/
│   │   └── schemas.py           # Pydantic 数据模型
│   ├── routes/
│   │   └── research.py          # SSE 流式 API 路由
│   ├── config.py                # 配置管理
│   ├── main.py                  # FastAPI 入口
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.vue              # 主布局
│   │   ├── components/
│   │   │   ├── ChatPanel.vue     # 聊天输入面板
│   │   │   ├── StepTimeline.vue  # 实时步骤时间线
│   │   │   ├── ReportView.vue    # 报告渲染
│   │   │   ├── HistoryList.vue   # 历史记录列表
│   │   │   └── HistoryDetail.vue # 历史记录详情
│   │   ├── composables/
│   │   │   └── useSSE.js         # SSE 组合式函数
│   │   ├── main.js
│   │   └── style.css
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── docs/                        # 设计文档
```

## 快速开始

### 1. 环境准备

- Python 3.10+
- Node.js 18+
- [SerpAPI](https://serpapi.com/) API Key（用于网络搜索）

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
SEARCH_API_KEY=your-serpapi-key
SEARCH_API_URL=https://serpapi.com/search
MAX_ROUNDS=10
MAX_TOKENS=8000
```

### 3. 启动后端

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

访问 `http://localhost:5173` 即可使用。

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/research` | 提交调研任务（SSE 流式返回） |
| `GET` | `/api/history` | 获取历史记录列表 |
| `GET` | `/api/history/{id}` | 获取某次调研详情 |
| `DELETE` | `/api/history/{id}` | 删除某次调研记录 |
| `GET` | `/health` | 健康检查 |

### SSE 事件类型

| 事件 | 说明 |
|------|------|
| `plan` | 调研计划（子问题列表） |
| `step` | ReAct 步骤（thought / action / observation） |
| `report` | 最终调研报告（Markdown） |
| `error` | 错误信息 |

## 工作原理

1. **生成计划**：LLM 将用户问题分解为多个子问题
2. **ReAct 循环**：每轮依次执行「思考 → 搜索 → 阅读网页 → 观察分析」
3. **记忆管理**：自动提取关键事实，避免上下文膨胀
4. **生成报告**：基于收集的信息生成结构化 Markdown 报告

## 技术栈

| 层 | 技术 |
|----|------|
| 后端框架 | FastAPI + Uvicorn |
| 流式传输 | Server-Sent Events (SSE) |
| LLM | OpenAI API 兼容接口 |
| 前端 | Vue 3 + Vite |
| Markdown 渲染 | marked |
| 数据库 | SQLite (aiosqlite) |
| 网页解析 | BeautifulSoup4 |
