<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { 
  Save, Loader2, Bot, GitBranch, Terminal, Shield, Play, Clock 
} from 'lucide-vue-next'

// --- 状态管理 ---
const config = ref({
  template_format: '',
  custom_rules: '',
  github_repo_url: '',
  ci_interval_minutes: 60
})

const loading = ref(false)
const saving = ref(false)
const runningCI = ref(false) // [新增] CI 运行状态
const status = ref({ msg: 'Connecting...', color: 'text-slate-500' })
const ciStatusInfo = ref({ last_run: 'Unknown', status: 'Idle' }) // [新增] CI 状态信息

// --- API 交互 ---
const loadConfig = async () => {
  loading.value = true
  try {
    // 并行获取配置和 CI 状态
    const [resConfig, resStatus] = await Promise.all([
      axios.get('/api/v1/config'),
      axios.get('/api/v1/ci/status')
    ])
    
    // 合并配置
    config.value = { 
      github_repo_url: '', 
      ci_interval_minutes: 60, 
      ...resConfig.data 
    }
    
    // 更新 CI 状态显示
    if (resStatus.data) {
      ciStatusInfo.value = resStatus.data
    }

    status.value = { msg: 'System Online', color: 'text-emerald-400' }
  } catch (error) {
    console.error(error)
    status.value = { msg: 'Server Disconnected', color: 'text-red-400' }
  } finally {
    loading.value = false
  }
}

const saveConfig = async () => {
  saving.value = true
  try {
    await axios.post('/api/v1/config', config.value)
    alert('✅ 配置已保存并生效')
  } catch (error) {
    alert('❌ 保存失败')
  } finally {
    saving.value = false
  }
}

// [新增] 手动触发 CI
const triggerCI = async () => {
  runningCI.value = true
  try {
    await axios.post('/api/v1/ci/run')
    alert('🚀 CI 流水线已触发，请查看日志')
    // 延迟刷新一下状态
    setTimeout(loadConfig, 2000)
  } catch (error) {
    alert('❌ 触发失败')
  } finally {
    runningCI.value = false
  }
}

onMounted(() => {
  loadConfig()
})
</script>

<template>
  <div class="min-h-screen bg-[#0f172a] text-slate-200 font-sans py-8 px-4 flex justify-center">
    
    <div class="w-full max-w-2xl space-y-6 pb-20">
      
      <div class="flex items-center justify-between px-2">
        <div class="flex items-center gap-3">
          <Shield class="w-8 h-8 text-blue-500" />
          <div>
            <h1 class="text-2xl font-bold text-white tracking-tight">Git-Guard</h1>
            <p class="text-xs font-mono opacity-60 flex items-center gap-2">
              <span :class="`w-2 h-2 rounded-full ${status.color.replace('text', 'bg')}`"></span>
              {{ status.msg }}
            </p>
          </div>
        </div>
        <a href="/api/v1/ci/status" target="_blank" class="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1 bg-slate-900 px-3 py-2 rounded-lg border border-slate-800 transition">
          <Terminal class="w-3 h-3" /> Raw Logs
        </a>
      </div>

      <div v-if="loading" class="py-20 flex justify-center text-slate-500">
        <Loader2 class="w-8 h-8 animate-spin" />
      </div>

      <form v-else @submit.prevent="saveConfig" class="space-y-6">
        
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
          <div class="flex items-center gap-2 mb-6 border-b border-slate-800 pb-4">
            <Bot class="w-5 h-5 text-purple-400" />
            <h2 class="font-semibold text-lg text-slate-100">AI Commit Rules</h2>
          </div>

          <div class="space-y-5">
            <div>
              <label class="block text-sm font-medium text-slate-400 mb-2">Template Format</label>
              <input 
                v-model="config.template_format"
                type="text" 
                class="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-sm focus:border-purple-500 focus:ring-1 focus:ring-purple-500 outline-none transition font-mono"
                placeholder="e.g. [<Module>] <Description>"
              >
            </div>

            <div>
              <label class="block text-sm font-medium text-slate-400 mb-2">Custom Instructions</label>
              <textarea 
                v-model="config.custom_rules"
                rows="4"
                class="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-sm focus:border-purple-500 focus:ring-1 focus:ring-purple-500 outline-none transition font-mono leading-relaxed"
                placeholder="Enter specific rules for the AI..."
              ></textarea>
            </div>
          </div>
        </div>

        <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
          <div class="flex items-center justify-between mb-6 border-b border-slate-800 pb-4">
            <div class="flex items-center gap-2">
              <GitBranch class="w-5 h-5 text-emerald-400" />
              <h2 class="font-semibold text-lg text-slate-100">Automated Pipeline</h2>
            </div>
            <div class="flex items-center gap-2 text-xs font-mono bg-slate-950 px-2 py-1 rounded border border-slate-800">
              <span :class="ciStatusInfo.status === 'Success' ? 'text-emerald-400' : 'text-slate-400'">
                {{ ciStatusInfo.status }}
              </span>
              <span class="text-slate-600">|</span>
              <span class="text-slate-500">{{ ciStatusInfo.last_run?.split(' ')[1] || 'Never' }}</span>
            </div>
          </div>

          <div class="space-y-5">
            <div>
              <label class="block text-sm font-medium text-slate-400 mb-2">Target Repository</label>
              <input 
                v-model="config.github_repo_url"
                type="url" 
                class="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-sm focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500 outline-none transition font-mono"
                placeholder="https://github.com/username/repo.git"
              >
            </div>

            <div>
              <label class="block text-sm font-medium text-slate-400 mb-2">
                Run Frequency: <span class="text-emerald-400 font-bold">{{ config.ci_interval_minutes }} min</span>
              </label>
              <input 
                v-model="config.ci_interval_minutes"
                type="range" 
                min="5" 
                max="1440" 
                step="5"
                class="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              >
              <div class="flex justify-between text-xs text-slate-600 mt-1">
                <span>5m</span>
                <span>24h</span>
              </div>
            </div>

            <div class="pt-4 border-t border-slate-800/50 flex items-center justify-between">
              <div class="text-xs text-slate-500 flex items-center gap-1">
                <Clock class="w-3 h-3" /> 
                Next run: {{ config.ci_interval_minutes }}m
              </div>
              
              <button 
                type="button"
                @click="triggerCI"
                :disabled="runningCI"
                class="flex items-center gap-2 px-4 py-2 bg-emerald-600/10 hover:bg-emerald-600/20 text-emerald-400 border border-emerald-600/30 rounded-lg transition-all text-sm font-bold disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Loader2 v-if="runningCI" class="w-4 h-4 animate-spin" />
                <Play v-else class="w-4 h-4 fill-current" />
                {{ runningCI ? 'Running...' : 'Run Pipeline Now' }}
              </button>
            </div>
          </div>
        </div>

        <div class="fixed bottom-0 left-0 right-0 p-4 bg-[#0f172a]/90 backdrop-blur border-t border-slate-800 z-50 flex justify-center">
          <button 
            type="submit" 
            :disabled="saving"
            class="w-full max-w-2xl bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white font-bold py-3.5 rounded-xl shadow-lg shadow-blue-900/30 transition-all flex items-center justify-center gap-2 active:scale-95"
          >
            <Loader2 v-if="saving" class="w-5 h-5 animate-spin" />
            <Save v-else class="w-5 h-5" />
            {{ saving ? 'Saving Changes...' : 'Save Configuration' }}
          </button>
        </div>

      </form>
      
    </div>
  </div>
</template>

<style scoped>
input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none;
  height: 16px;
  width: 16px;
  border-radius: 50%;
  background: #10b981;
  margin-top: -6px;
}
input[type=range]::-webkit-slider-runnable-track {
  height: 4px;
  background: #1e293b;
}
</style>