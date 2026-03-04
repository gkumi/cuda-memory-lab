# CUDA Host/Device Memory Lab (Paged, Pinned, Mapped)

This repository contains a CUDA programming lab focused on **correctly allocating memory**, **transferring data between host and device**, **launching kernels**, and **debugging common CUDA memory mistakes**.

There are multiple parts that cover:
- **Pageable (paged) host memory** vs **Pinned host memory**
- **Host→Device copies** and **Device→Host copies**
- **Unified (managed) memory**
- **Mapped (zero-copy) host memory**

---

## Project Structure

| File | Purpose |
|------|---------|
| `memory_allocation.cu` | Part 1–2: Host/device memory allocation (paged, pinned, unified) |
| `memory_copy.cu` | Part 3–4: Correct host→device copies + device→host copies |
| `broken_paged_pinned_memory_allocation.cu` | Part 5: Debug/fix paged + pinned allocation/copy issues |
| `broken_mapped_memory_allocation.cu` | Part 6: Debug/fix mapped (zero-copy) memory implementation |
| `run.sh` | Builds and runs all parts (generates output files) |
| `validate_output.py` | Optional validator script (if provided by course) |
| `test_data.csv`, `test_float_data.csv` | Input data for testing |
| `Makefile` | Build targets for each part |

---

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc available)
- Linux environment (Coursera lab or local machine)

---

## How to Build

### Build individual parts
```bash
make build-memory-allocation
make build-memory-copy
make build-broken-paged-pinned-memory-allocation
make build-broken-mapped-memory-allocation
Build and run everything
./run.sh

Outputs will be written to files like:

output-<PartID>.txt

Assignment Workflow (Parts 1–6)

Follow this order to complete the lab:

Complete memory allocation logic in memory_allocation.cu
Look for blocks marked:
// FILL IN HOST AND DEVICE MEMORY ALLOCATION CODE

Click/Run Build Parts 1 and 2 (or make build-memory-allocation).

Complete host→device copy logic in memory_copy.cu
Look for blocks marked:
// FILL IN HOST AND DEVICE MEMORY COPY CODE

Click/Run Build Parts 3 and 4 (or make build-memory-copy).

Fix the broken program in broken_paged_pinned_memory_allocation.cu
There are no hints—use debugging and CUDA error checks.

Click/Run Build Part 5.

Fix the broken program in broken_mapped_memory_allocation.cu
Focus on correct mapped memory allocation, device pointer mapping, and cleanup.

Click/Run Build Part 6.

When everything works as expected, submit Parts 1–6 via your course platform.

Notes & Tips

Always check CUDA return codes (cudaError_t) when allocating/copying.

If using mapped memory, avoid cudaMemcpy and do not cudaFree() mapped pointers.

For unified memory, you may need cudaDeviceSynchronize() before reading results on the CPU.

Keep binaries and output files out of Git by using .gitignore (already included).

What You Learn

By completing this lab you will be able to:

Correctly allocate CUDA memory across multiple strategies (paged, pinned, unified, mapped)

Transfer data safely and efficiently between host and device

Debug CUDA programs that “work” but produce incorrect results

Understand when to use each memory type and the trade-offs involved
