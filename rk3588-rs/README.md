# rk3588-rs

Rust bindings for RK3588 NPU (Neural Processing Unit) hardware access.

This library is a Rust translation of the [rk3588-npu](https://github.com/starry-mix-rk3588/rknpu2-rslab/tree/main/rk3588-npu) C library, providing safe and idiomatic Rust interfaces for programming the RK3588 NPU directly.

## Features

- **Hardware Register Definitions**: Complete register definitions for CNA, DPU, CORE, and PC units
- **Memory Management**: Safe abstractions for NPU memory allocation and management
- **Matrix Multiplication**: High-performance FP16 and INT8 matrix multiplication operations
- **IOCTL Bindings**: Complete bindings for RKNPU kernel driver ioctls
- **Safe Abstractions**: Rust wrappers providing memory safety and type safety

## Modules

- `hw`: Hardware register definitions and constants
- `ioctl`: IOCTL structures and bindings for kernel driver communication
- `cna`: Convolution Neural Accelerator (CNA) descriptor structures
- `dpu`: Data Processing Unit (DPU) descriptor structures
- `matmul`: Matrix multiplication functions for FP16 and INT8
- `interface`: High-level NPU device and memory management interfaces

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
rk3588-rs = { path = "path/to/rk3588-rs" }
```

### Basic Example

```rust
use rk3588_rs::{NpuDevice, MatmulParams, gen_matmul_fp16};

fn main() -> std::io::Result<()> {
    // Open NPU device
    let npu = NpuDevice::open()?;
    
    // Reset NPU
    npu.reset()?;
    
    // Allocate memory for matrices
    let input_size = 64 * 64 * 2; // M x K, FP16
    let weights_size = 64 * 64 * 2; // K x N, FP16
    let output_size = 64 * 64 * 4; // M x N, FP32
    
    let mut input_mem = npu.mem_allocate(input_size, 0)?;
    let mut weights_mem = npu.mem_allocate(weights_size, 0)?;
    let mut output_mem = npu.mem_allocate(output_size, 0)?;
    let mut task_mem = npu.mem_allocate(112 * 8, 0)?; // 112 u64 values
    
    // Initialize input and weights data
    // ... (fill input_mem and weights_mem with data)
    
    // Configure matrix multiplication
    let mut params = MatmulParams {
        m: 64,
        k: 64,
        n: 64,
        input_dma: input_mem.dma_addr() as u32,
        weights_dma: weights_mem.dma_addr() as u32,
        output_dma: output_mem.dma_addr() as u32,
        tasks: task_mem.as_ptr() as *mut u64,
        fp32tofp16: 0, // Output as FP32
    };
    
    // Generate matrix multiplication task
    gen_matmul_fp16(&mut params)?;
    
    // Submit task to NPU
    // ... (submit using RknpuSubmit structure)
    
    Ok(())
}
```

### Legacy C-style API

For compatibility with existing C code, legacy functions are also available:

```rust
use rk3588_rs::{npu_open, npu_close, npu_reset, mem_allocate, mem_destroy};

unsafe {
    let fd = npu_open()?;
    npu_reset(fd)?;
    
    let size = 1024;
    let (ptr, dma_addr, obj_addr, handle) = mem_allocate(fd, size, 0)?;
    
    // Use the memory...
    
    mem_destroy(fd, handle, obj_addr)?;
    npu_close(fd)?;
}
```

## Safety

This library interacts directly with hardware and kernel drivers. While the high-level `NpuDevice` and `NpuMemory` types provide safe abstractions, some operations (like raw pointer manipulation for tasks) require `unsafe` code. Users should ensure:

- Valid memory sizes and alignments
- Proper synchronization when submitting tasks
- Correct DMA address usage

## Hardware Requirements

- RK3588 SoC with NPU support
- Linux kernel with RKNPU driver (`/dev/dri/card1`)
- Appropriate permissions to access NPU device

## License

This library is licensed under GPL-3.0-or-later, matching the original C library.

Original Copyright (C) 2024 Jasbir Matharu <jasjnuk@gmail.com>

## Credits

This is a Rust translation of the excellent [rk3588-npu](https://github.com/starry-mix-rk3588/rknpu2-rslab) C library by Jasbir Matharu. All hardware knowledge and algorithms are derived from that work.

## Documentation

For detailed hardware documentation and programming guide, please refer to:
- RK3588 Technical Reference Manual (TRM)
- Original C library documentation
- NPU programming examples in the repository
