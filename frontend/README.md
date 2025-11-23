# DeepSeek-style Chat Frontend (React + TypeScript + Vite)

This is a standalone frontend built to be dropped into your repository (recommended under `/frontend`).

## Features
- Append-style chat UI (user messages appended)
- Assistant responses show `context` (思考) below the answer in small text
- `POST /api/user/ask` is used to send questions
- Small, dependency-light setup (axios, uuid)

## Quick start
1. Put this folder under your project: `/frontend`
2. Install dependencies:
```bash
npm install
# or
yarn
```
3. Run dev:
```bash
npm run dev
```
The Vite dev server proxies `/api` to `http://localhost:8000` (edit `vite.config.ts` if your backend is elsewhere).

## Notes
- Styles are simple CSS (no Tailwind) for easy integration.
- To enable streaming (SSE) or file upload, adapt `src/services/chat.ts` and components.
