# llama-cpp-python 0.3.9 forces Metal on Apple platforms in its top-level build.
# Reapply CPU settings at every nested project() before llama.cpp adds backends.
set(GGML_METAL OFF CACHE BOOL "CPU-only cafe-agent build" FORCE)
set(GGML_METAL_EMBED_LIBRARY OFF CACHE BOOL "No Metal shader library" FORCE)
set(GGML_CUDA OFF CACHE BOOL "No CUDA backend" FORCE)
