<script setup lang="ts">
import type { HTMLAttributes } from 'vue'
import { cn } from '~/lib/utils'

const props = defineProps<{
  variant?: 'default' | 'outline' | 'success' | 'warning' | 'error'
  class?: HTMLAttributes['class']
}>()
</script>

<template>
  <span
    :class="cn(
      'inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium transition-colors',
      {
        'bg-primary/10 text-primary': variant === 'default' || !variant,
        'border border-border text-muted-foreground': variant === 'outline',
        'bg-emerald-50 text-emerald-700': variant === 'success',
        'bg-amber-50 text-amber-700': variant === 'warning',
        'bg-red-50 text-red-700': variant === 'error',
      },
      props.class,
    )"
  >
    <slot />
  </span>
</template>
