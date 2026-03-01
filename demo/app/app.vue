<script setup lang="ts">
import {
  Eye,
  Zap,
  RotateCcw,
  Loader2,
  Check,
  AlertTriangle,
  Pencil,
  ClipboardPaste,
  ImagePlus,
} from 'lucide-vue-next'

const {
  isInitialized,
  isInitializing,
  initError,
  processError,
  results,
  tokenCount,
  initialize,
  detect,
  clear,
} = useMacula()

// Form state
const wasmUrl = ref('https://raw.githubusercontent.com/xirf/macula/main/zig-out/bin/ocr_wasm.wasm')
const artifactUrl = ref('https://raw.githubusercontent.com/xirf/macula/main/data/ocr_detector.bin')
const threshold = ref('5.0')
const inputText = ref('Teh quicK brovvn fox jumps ouer teh lazy d0g')
const imageBlob = ref<Blob | null>(null)
const imagePreviewUrl = ref<string | null>(null)
const ocrStatus = ref('')
const ocrLoading = ref(false)
const showSettings = ref(false)

// Count stats
const validCount = computed(() => results.value.filter(r => r.status === 'valid').length)
const correctedCount = computed(() => results.value.filter(r => r.status === 'corrected').length)
const errorCount = computed(() => results.value.filter(r => r.status === 'error_detected').length)

async function onInit() {
  await initialize(wasmUrl.value, artifactUrl.value, parseFloat(threshold.value) || 5.0)
}

function onDetect() {
  if (!inputText.value.trim()) return
  detect(inputText.value)
}

function onClear() {
  inputText.value = ''
  clear()
  ocrStatus.value = ''
  imageBlob.value = null
  if (imagePreviewUrl.value) {
    URL.revokeObjectURL(imagePreviewUrl.value)
    imagePreviewUrl.value = null
  }
}

function setPreviewImage(blob: Blob) {
  imageBlob.value = blob
  if (imagePreviewUrl.value) URL.revokeObjectURL(imagePreviewUrl.value)
  imagePreviewUrl.value = URL.createObjectURL(blob)
}

function onPaste(e: ClipboardEvent) {
  const items = e.clipboardData?.items || []
  for (const item of items) {
    if (item.type.startsWith('image/')) {
      const blob = item.getAsFile()
      if (blob) {
        setPreviewImage(blob)
        ocrStatus.value = 'Image pasted — click Extract Text.'
        e.preventDefault()
      }
      return
    }
  }
}

function onImageFile(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (file) {
    setPreviewImage(file)
    ocrStatus.value = 'Image selected — click Extract Text.'
  }
}

async function onExtractOcr() {
  if (!imageBlob.value) {
    ocrStatus.value = 'Paste or select an image first.'
    return
  }

  ocrLoading.value = true
  ocrStatus.value = 'Loading OCR engine...'

  try {
    // Dynamically load tesseract.js
    if (!(window as any).Tesseract?.recognize) {
      await new Promise<void>((resolve, reject) => {
        const script = document.createElement('script')
        script.src = 'https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js'
        script.onload = () => resolve()
        script.onerror = () => reject(new Error('Failed to load tesseract.js'))
        document.head.appendChild(script)
      })
    }

    ocrStatus.value = 'Extracting text...'
    const { data } = await (window as any).Tesseract.recognize(imageBlob.value, 'eng')
    const text = (data?.text || '').trim()
    inputText.value = text
    ocrStatus.value = text.length > 0
      ? `Done — ${text.length} characters extracted.`
      : 'No text found in image.'
  }
  catch (err: any) {
    ocrStatus.value = `OCR failed: ${err.message}`
  }
  finally {
    ocrLoading.value = false
  }
}

function statusBadgeVariant(status: string) {
  switch (status) {
    case 'valid': return 'success'
    case 'corrected': return 'warning'
    case 'error_detected': return 'error'
    default: return 'outline'
  }
}

function statusLabel(status: string) {
  switch (status) {
    case 'valid': return 'Valid'
    case 'corrected': return 'Corrected'
    case 'error_detected': return 'Error'
    default: return status
  }
}

// Auto-init on mount — total payload is only ~800 KB (680 KB WASM + 118 KB model)
onMounted(() => {
  document.addEventListener('paste', onPaste as any)
  onInit()
})

onUnmounted(() => {
  document.removeEventListener('paste', onPaste as any)
  if (imagePreviewUrl.value) URL.revokeObjectURL(imagePreviewUrl.value)
})
</script>

<template>
  <div class="min-h-screen bg-muted">
    <!-- Header -->
    <header class="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur-sm">
      <div class="mx-auto flex h-14 max-w-4xl items-center justify-between px-4">
        <div class="flex items-center gap-2.5">
          <Eye class="h-5 w-5 text-primary" :stroke-width="2.5" />
          <span class="text-lg font-bold tracking-tight">Macula</span>
        </div>
        <span class="hidden text-sm text-muted-foreground sm:inline">
          OCR Error Detector · Client-side WASM
        </span>
      </div>
    </header>

    <main class="mx-auto max-w-4xl space-y-4 px-4 py-6">
      <!--  Initialize — only show when loading/error/settings -->
      <UiCard v-if="isInitializing || initError || showSettings || !isInitialized" class="p-5">
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-sm font-semibold">Model</h2>
            <p class="mt-0.5 text-xs text-muted-foreground">
              {{ isInitialized ? 'Ready to detect.' : 'Load the WASM engine and dictionary artifact.' }}
            </p>
          </div>
          <div class="flex items-center gap-2">
            <button
              class="text-xs text-muted-foreground underline-offset-2 hover:underline cursor-pointer"
              @click="showSettings = !showSettings"
            >
              {{ showSettings ? 'Hide' : 'Settings' }}
            </button>
            <UiButton
              :disabled="isInitializing"
              size="sm"
              @click="onInit"
            >
              <Loader2 v-if="isInitializing" class="h-3.5 w-3.5 animate-spin" />
              <Zap v-else class="h-3.5 w-3.5" />
              {{ isInitializing ? 'Loading…' : isInitialized ? 'Reinitialize' : 'Initialize' }}
            </UiButton>
          </div>
        </div>

        <!-- Collapsible settings -->
        <Transition name="slide">
          <div v-if="showSettings" class="mt-4 grid gap-3 sm:grid-cols-3">
            <div>
              <label class="mb-1 block text-xs font-medium text-muted-foreground">WASM URL</label>
              <UiInput v-model="wasmUrl" placeholder="https://raw.githubusercontent.com/.../ocr_wasm.wasm" class="text-xs" />
            </div>
            <div>
              <label class="mb-1 block text-xs font-medium text-muted-foreground">Artifact URL</label>
              <UiInput v-model="artifactUrl" placeholder="https://raw.githubusercontent.com/.../ocr_detector.bin" class="text-xs" />
            </div>
            <div>
              <label class="mb-1 block text-xs font-medium text-muted-foreground">Threshold</label>
              <UiInput v-model="threshold" type="number" step="0.1" class="text-xs" />
            </div>
          </div>
        </Transition>

        <p v-if="initError" class="mt-3 text-xs text-destructive">{{ initError }}</p>
      </UiCard>

      <!-- Image OCR -->
      <UiCard class="p-5">
        <h2 class="text-sm font-semibold">Image Input</h2>
        <p class="mt-0.5 text-xs text-muted-foreground">Paste an image (Ctrl+V) or pick a file to extract text via Tesseract.js.</p>

        <div class="mt-3 flex flex-col gap-3 sm:flex-row sm:items-end">
          <div class="flex gap-2">
            <label
              class="inline-flex h-9 cursor-pointer items-center gap-1.5 rounded-lg border border-dashed border-border px-3 text-xs text-muted-foreground transition-colors hover:border-primary hover:text-primary"
            >
              <ImagePlus class="h-3.5 w-3.5" />
              Choose image
              <input type="file" accept="image/*" class="hidden" @change="onImageFile" />
            </label>
            <UiButton variant="outline" size="sm" :disabled="!imageBlob || ocrLoading" @click="onExtractOcr">
              <Loader2 v-if="ocrLoading" class="h-3.5 w-3.5 animate-spin" />
              <ClipboardPaste v-else class="h-3.5 w-3.5" />
              Extract Text
            </UiButton>
          </div>
          <span v-if="ocrStatus" class="text-xs text-muted-foreground">{{ ocrStatus }}</span>
        </div>

        <img
          v-if="imagePreviewUrl"
          :src="imagePreviewUrl"
          alt="Uploaded image preview"
          class="mt-3 max-h-48 rounded-lg border border-border object-contain"
        />
      </UiCard>

      <!-- Text Input + Run -->
      <UiCard class="p-5">
        <div class="flex items-center justify-between">
          <h2 class="text-sm font-semibold">Text</h2>
          <div class="flex gap-2">
            <UiButton variant="ghost" size="sm" @click="onClear">
              <RotateCcw class="h-3.5 w-3.5" />
              Clear
            </UiButton>
            <UiButton
              size="sm"
              :disabled="!isInitialized || !inputText.trim()"
              @click="onDetect"
            >
              <Zap class="h-3.5 w-3.5" />
              Run Detection
            </UiButton>
          </div>
        </div>
        <UiTextarea
          v-model="inputText"
          placeholder="Paste or type OCR output here…"
          class="mt-3 font-mono text-sm"
          :rows="4"
        />
      </UiCard>

      <!-- Results -->
      <UiCard v-if="results.length > 0 || processError" class="overflow-hidden">
        <div class="border-b border-border px-5 py-4">
          <div class="flex items-center justify-between">
            <h2 class="text-sm font-semibold">Results</h2>
            <span class="text-xs text-muted-foreground">
              {{ tokenCount }} token{{ tokenCount !== 1 ? 's' : '' }} processed
            </span>
          </div>

          <!-- Stats row -->
          <div v-if="results.length > 0" class="mt-2 flex gap-4 text-xs">
            <span class="flex items-center gap-1 text-emerald-600">
              <Check class="h-3 w-3" /> {{ validCount }} valid
            </span>
            <span class="flex items-center gap-1 text-amber-600">
              <Pencil class="h-3 w-3" /> {{ correctedCount }} corrected
            </span>
            <span class="flex items-center gap-1 text-red-600">
              <AlertTriangle class="h-3 w-3" /> {{ errorCount }} errors
            </span>
          </div>
        </div>

        <p v-if="processError" class="px-5 py-3 text-xs text-destructive">{{ processError }}</p>

        <div v-if="results.length > 0" class="overflow-x-auto">
          <table class="w-full text-left text-sm">
            <thead>
              <tr class="border-b border-border text-xs text-muted-foreground">
                <th class="px-5 py-2 font-medium">#</th>
                <th class="px-5 py-2 font-medium">Token</th>
                <th class="px-5 py-2 font-medium">Status</th>
                <th class="px-5 py-2 font-medium">Correction</th>
                <th class="px-5 py-2 font-medium text-right">Confidence</th>
                <th class="px-5 py-2 font-medium text-right">Span</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(row, i) in results"
                :key="i"
                class="border-b border-border last:border-0 transition-colors hover:bg-muted/50"
              >
                <td class="px-5 py-2.5 text-xs text-muted-foreground tabular-nums">{{ i + 1 }}</td>
                <td class="px-5 py-2.5 font-mono text-xs font-medium">{{ row.token }}</td>
                <td class="px-5 py-2.5">
                  <UiBadge :variant="statusBadgeVariant(row.status) as any">
                    {{ statusLabel(row.status) }}
                  </UiBadge>
                </td>
                <td class="px-5 py-2.5 font-mono text-xs">
                  {{ row.correction ?? '—' }}
                </td>
                <td class="px-5 py-2.5 text-right font-mono text-xs tabular-nums">
                  {{ row.confidence.toFixed(4) }}
                </td>
                <td class="px-5 py-2.5 text-right text-xs text-muted-foreground tabular-nums">
                  {{ row.start }}–{{ row.start + row.len }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </UiCard>
    </main>

    <!-- Footer -->
    <footer class="border-t border-border py-6 text-center text-xs text-muted-foreground">
      Macula — Zero-server OCR error detection. Powered by Zig + WebAssembly.
    </footer>
  </div>
</template>

<style>
.slide-enter-active,
.slide-leave-active {
  transition: all 0.2s ease;
  overflow: hidden;
}
.slide-enter-from,
.slide-leave-to {
  opacity: 0;
  max-height: 0;
  margin-top: 0;
}
.slide-enter-to,
.slide-leave-from {
  opacity: 1;
  max-height: 200px;
}
</style>
