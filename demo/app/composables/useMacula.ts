/**
 * Macula WASM OCR Engine composable.
 * Manages the lifecycle of the WebAssembly module and provides
 * reactive state for the Vue UI.
 */

export interface DetectionRow {
    token: string
    status: 'valid' | 'corrected' | 'error_detected'
    correction: string | null
    confidence: number
    start: number
    len: number
}

const decoder = new TextDecoder()
const encoder = new TextEncoder()

let wasmInstance: WebAssembly.Instance | null = null
let wasmExports: Record<string, any> | null = null

function getMemory(): WebAssembly.Memory {
    return wasmExports!.memory
}

function ptrLenToString(ptr: number, len: number): string {
    if (!ptr || !len) return ''
    const bytes = new Uint8Array(getMemory().buffer, ptr, len)
    return decoder.decode(bytes)
}

function getLastError(): string {
    const ptr = wasmExports!.ocr_last_error_ptr()
    const len = wasmExports!.ocr_last_error_len()
    return ptrLenToString(ptr, len)
}

function allocBytes(data: Uint8Array): { ptr: number; len: number } {
    const ptr = wasmExports!.wasm_alloc(data.length)
    if (!ptr) throw new Error('wasm_alloc failed')
    const dst = new Uint8Array(getMemory().buffer, ptr, data.length)
    dst.set(data)
    return { ptr, len: data.length }
}

function freeBytes(ptr: number, len: number) {
    if (ptr && len) wasmExports!.wasm_free(ptr, len)
}

async function fetchBytes(url: string): Promise<Uint8Array> {
    const bust = `${url}${url.includes('?') ? '&' : '?'}v=${Date.now()}`
    const res = await fetch(bust, { cache: 'no-store' })
    if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`)
    return new Uint8Array(await res.arrayBuffer())
}

export function useMacula() {
    const isInitialized = useState('macula-init', () => false)
    const isInitializing = useState('macula-loading', () => false)
    const isProcessing = useState('macula-processing', () => false)
    const initError = useState<string | null>('macula-init-error', () => null)
    const processError = useState<string | null>('macula-process-error', () => null)
    const results = useState<DetectionRow[]>('macula-results', () => [])
    const tokenCount = useState('macula-token-count', () => 0)

    async function initialize(
        wasmUrl: string,
        artifactUrl: string,
        threshold: number,
    ) {
        isInitializing.value = true
        initError.value = null

        try {
            const wasmBytes = await fetchBytes(wasmUrl)
            const wasmModule = await WebAssembly.instantiate(wasmBytes, {})
            wasmInstance = wasmModule.instance
            wasmExports = wasmInstance.exports as Record<string, any>

            const artifactBytes = await fetchBytes(artifactUrl)
            const modelMem = allocBytes(artifactBytes)

            const rc = wasmExports.ocr_init(modelMem.ptr, modelMem.len, threshold)
            freeBytes(modelMem.ptr, modelMem.len)

            if (rc !== 0) {
                throw new Error(getLastError() || `ocr_init failed (${rc})`)
            }

            isInitialized.value = true
        }
        catch (err: any) {
            initError.value = err.message
            isInitialized.value = false
        }
        finally {
            isInitializing.value = false
        }
    }

    function detect(text: string) {
        processError.value = null
        results.value = []
        tokenCount.value = 0

        if (!wasmExports) {
            processError.value = 'Model is not initialized.'
            return
        }

        try {
            const textBytes = encoder.encode(text)
            const inputMem = allocBytes(textBytes)
            const count = wasmExports.ocr_process(inputMem.ptr, inputMem.len)
            freeBytes(inputMem.ptr, inputMem.len)

            if (count < 0) {
                throw new Error(getLastError() || `ocr_process failed (${count})`)
            }

            const outPtr = wasmExports.ocr_output_ptr()
            const outLen = wasmExports.ocr_output_len()
            const json = ptrLenToString(outPtr, outLen)
            const rows: DetectionRow[] = JSON.parse(json)

            results.value = rows
            tokenCount.value = rows.length
        }
        catch (err: any) {
            processError.value = err.message
        }
    }

    function clear() {
        results.value = []
        tokenCount.value = 0
        processError.value = null
    }

    return {
        isInitialized,
        isInitializing,
        isProcessing,
        initError,
        processError,
        results,
        tokenCount,
        initialize,
        detect,
        clear,
    }
}
