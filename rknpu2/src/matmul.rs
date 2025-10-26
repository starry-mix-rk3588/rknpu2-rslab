/*
 * Matrix multiplication test using rk3588-rs library
 *
 * This is a Rust translation of matmul_4_36_16.c test
 * Performs a 4x36 * 36x16 matrix multiplication using RK3588 NPU
 */

use half::f16;
use rk3588_rs::{
    MatmulParams, RKNPU_JOB_BLOCK, RKNPU_JOB_PC, RKNPU_JOB_PINGPONG, RKNPU_MEM_KERNEL_MAPPING,
    RKNPU_PC_DATA_EXTRA_AMOUNT, RknpuSubcoreTask, RknpuSubmit, RknpuTask, feature_data,
    gen_matmul_fp16, weight_fp16,
};
use std::io;

// Test data from test-mul-mat.cpp (ggml)
// Matrix A (4 x 36)
const MATRIX_A: [f32; 144] = [
    2.0, 9.0, 2.0, 10.0, 6.0, 4.0, 3.0, 6.0, 3.0, 6.0, 9.0, 7.0, 8.0, 8.0, 3.0, 3.0, 10.0, 5.0,
    2.0, 10.0, 7.0, 10.0, 9.0, 3.0, 6.0, 6.0, 5.0, 10.0, 2.0, 3.0, 6.0, 1.0, 9.0, 4.0, 10.0, 4.0,
    10.0, 7.0, 8.0, 10.0, 10.0, 8.0, 7.0, 10.0, 4.0, 6.0, 8.0, 7.0, 7.0, 6.0, 9.0, 3.0, 6.0, 5.0,
    5.0, 2.0, 7.0, 2.0, 7.0, 4.0, 4.0, 6.0, 6.0, 4.0, 3.0, 9.0, 3.0, 6.0, 4.0, 7.0, 2.0, 9.0, 7.0,
    3.0, 2.0, 5.0, 7.0, 3.0, 10.0, 2.0, 6.0, 1.0, 4.0, 7.0, 5.0, 10.0, 3.0, 10.0, 4.0, 5.0, 5.0,
    1.0, 6.0, 10.0, 7.0, 4.0, 5.0, 3.0, 9.0, 9.0, 8.0, 6.0, 9.0, 2.0, 3.0, 6.0, 8.0, 5.0, 5.0, 5.0,
    5.0, 5.0, 3.0, 10.0, 4.0, 1.0, 8.0, 8.0, 9.0, 8.0, 4.0, 1.0, 4.0, 9.0, 3.0, 6.0, 3.0, 1.0, 4.0,
    8.0, 3.0, 10.0, 8.0, 6.0, 4.0, 5.0, 4.0, 3.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0,
];

// Matrix B (16 x 36)
const MATRIX_B: [f32; 576] = [
    9.0, 7.0, 1.0, 3.0, 5.0, 9.0, 7.0, 6.0, 1.0, 10.0, 1.0, 1.0, 7.0, 2.0, 4.0, 9.0, 10.0, 4.0,
    5.0, 5.0, 7.0, 1.0, 7.0, 7.0, 2.0, 9.0, 5.0, 10.0, 7.0, 4.0, 8.0, 9.0, 9.0, 3.0, 10.0, 2.0,
    4.0, 6.0, 10.0, 9.0, 5.0, 1.0, 8.0, 7.0, 4.0, 7.0, 2.0, 6.0, 5.0, 3.0, 1.0, 10.0, 8.0, 4.0,
    8.0, 3.0, 7.0, 1.0, 2.0, 7.0, 6.0, 8.0, 6.0, 5.0, 2.0, 3.0, 1.0, 1.0, 2.0, 5.0, 7.0, 1.0, 8.0,
    2.0, 8.0, 8.0, 8.0, 8.0, 4.0, 4.0, 6.0, 10.0, 10.0, 9.0, 2.0, 9.0, 3.0, 7.0, 7.0, 1.0, 4.0,
    9.0, 1.0, 2.0, 3.0, 6.0, 1.0, 10.0, 5.0, 8.0, 9.0, 4.0, 6.0, 2.0, 3.0, 1.0, 2.0, 7.0, 5.0, 1.0,
    7.0, 2.0, 9.0, 10.0, 9.0, 5.0, 2.0, 5.0, 4.0, 10.0, 9.0, 9.0, 1.0, 9.0, 8.0, 8.0, 9.0, 4.0,
    9.0, 4.0, 8.0, 2.0, 1.0, 8.0, 4.0, 5.0, 10.0, 7.0, 6.0, 2.0, 1.0, 10.0, 10.0, 7.0, 9.0, 4.0,
    5.0, 9.0, 5.0, 10.0, 10.0, 3.0, 6.0, 6.0, 4.0, 4.0, 4.0, 8.0, 5.0, 4.0, 9.0, 1.0, 9.0, 9.0,
    1.0, 7.0, 9.0, 2.0, 10.0, 9.0, 10.0, 8.0, 3.0, 3.0, 9.0, 3.0, 9.0, 10.0, 1.0, 8.0, 9.0, 2.0,
    6.0, 9.0, 7.0, 2.0, 3.0, 5.0, 3.0, 6.0, 9.0, 7.0, 3.0, 7.0, 6.0, 4.0, 10.0, 3.0, 5.0, 7.0, 2.0,
    9.0, 3.0, 2.0, 2.0, 10.0, 8.0, 7.0, 3.0, 10.0, 6.0, 3.0, 1.0, 1.0, 4.0, 10.0, 2.0, 9.0, 2.0,
    10.0, 6.0, 4.0, 3.0, 6.0, 3.0, 6.0, 9.0, 7.0, 8.0, 8.0, 3.0, 3.0, 10.0, 5.0, 2.0, 10.0, 7.0,
    10.0, 9.0, 3.0, 6.0, 6.0, 5.0, 10.0, 2.0, 3.0, 6.0, 1.0, 9.0, 4.0, 10.0, 4.0, 10.0, 7.0, 8.0,
    10.0, 10.0, 8.0, 7.0, 10.0, 4.0, 6.0, 8.0, 7.0, 7.0, 6.0, 9.0, 3.0, 6.0, 5.0, 5.0, 2.0, 7.0,
    2.0, 7.0, 4.0, 4.0, 6.0, 6.0, 4.0, 3.0, 9.0, 3.0, 6.0, 4.0, 7.0, 2.0, 9.0, 7.0, 3.0, 2.0, 5.0,
    7.0, 3.0, 10.0, 2.0, 6.0, 1.0, 4.0, 7.0, 5.0, 10.0, 3.0, 10.0, 4.0, 5.0, 5.0, 1.0, 6.0, 10.0,
    7.0, 4.0, 5.0, 3.0, 9.0, 9.0, 8.0, 6.0, 9.0, 2.0, 3.0, 6.0, 8.0, 5.0, 5.0, 5.0, 5.0, 5.0, 3.0,
    10.0, 4.0, 1.0, 8.0, 8.0, 9.0, 8.0, 4.0, 1.0, 4.0, 9.0, 3.0, 6.0, 3.0, 1.0, 4.0, 8.0, 3.0,
    10.0, 8.0, 6.0, 4.0, 5.0, 4.0, 3.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 6.0, 2.0, 3.0, 3.0, 3.0, 7.0,
    5.0, 1.0, 8.0, 1.0, 4.0, 5.0, 1.0, 1.0, 6.0, 4.0, 2.0, 1.0, 7.0, 8.0, 6.0, 1.0, 1.0, 5.0, 6.0,
    5.0, 10.0, 6.0, 7.0, 5.0, 9.0, 3.0, 2.0, 7.0, 9.0, 4.0, 2.0, 5.0, 9.0, 5.0, 10.0, 3.0, 1.0,
    8.0, 1.0, 7.0, 1.0, 8.0, 1.0, 6.0, 7.0, 8.0, 4.0, 9.0, 5.0, 10.0, 3.0, 7.0, 6.0, 8.0, 8.0, 5.0,
    6.0, 8.0, 10.0, 9.0, 4.0, 1.0, 3.0, 3.0, 4.0, 7.0, 8.0, 2.0, 6.0, 6.0, 5.0, 1.0, 3.0, 7.0, 1.0,
    7.0, 2.0, 2.0, 2.0, 8.0, 4.0, 1.0, 1.0, 5.0, 9.0, 4.0, 1.0, 2.0, 3.0, 10.0, 1.0, 4.0, 9.0, 9.0,
    6.0, 8.0, 8.0, 1.0, 9.0, 10.0, 4.0, 1.0, 8.0, 5.0, 8.0, 9.0, 4.0, 8.0, 2.0, 1.0, 1.0, 9.0, 4.0,
    5.0, 6.0, 1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 1.0, 4.0, 6.0, 7.0, 7.0, 7.0, 8.0, 7.0, 8.0, 8.0, 2.0,
    10.0, 2.0, 7.0, 3.0, 8.0, 3.0, 8.0, 7.0, 6.0, 2.0, 4.0, 10.0, 10.0, 6.0, 10.0, 3.0, 7.0, 6.0,
    4.0, 3.0, 5.0, 5.0, 5.0, 3.0, 8.0, 10.0, 3.0, 4.0, 8.0, 4.0, 2.0, 6.0, 8.0, 9.0, 6.0, 9.0, 4.0,
    3.0, 5.0, 2.0, 2.0, 6.0, 10.0, 6.0, 2.0, 1.0, 7.0, 5.0, 6.0, 4.0, 1.0, 9.0, 10.0, 2.0, 4.0,
    5.0, 8.0, 5.0, 7.0, 4.0, 7.0, 6.0, 3.0, 9.0, 2.0, 1.0, 4.0, 2.0, 6.0, 6.0, 3.0, 3.0, 2.0, 8.0,
    5.0, 9.0, 3.0, 4.0,
];

// Expected result matrix C (4 x 16)
const EXPECTED_RESULT: [f32; 64] = [
    1224.0, 1023.0, 1158.0, 1259.0, 1359.0, 1194.0, 1535.0, 1247.0, 1185.0, 1029.0, 889.0, 1182.0,
    955.0, 1179.0, 1147.0, 1048.0, 1216.0, 1087.0, 1239.0, 1361.0, 1392.0, 1260.0, 1247.0, 1563.0,
    1167.0, 1052.0, 942.0, 1214.0, 1045.0, 1134.0, 1264.0, 1126.0, 1125.0, 966.0, 1079.0, 1333.0,
    1287.0, 1101.0, 1185.0, 1167.0, 1368.0, 990.0, 967.0, 1121.0, 971.0, 1086.0, 1130.0, 980.0,
    999.0, 902.0, 1020.0, 1056.0, 1076.0, 929.0, 1029.0, 1052.0, 990.0, 1108.0, 823.0, 989.0,
    759.0, 1041.0, 1003.0, 870.0,
];

pub fn run_matmul_test() -> io::Result<()> {
    const M: usize = 4;
    const K: usize = 36;
    const N: usize = 16;

    println!("Starting RK3588 NPU matrix multiplication test");
    println!(
        "Matrix dimensions: A({}x{}) * B({}x{}) = C({}x{})",
        M, K, K, N, M, N
    );

    // Open NPU device
    let npu = rk3588_rs::NpuDevice::open()?;

    // Allocate memory using new Rust API
    let mut regcmd_mem = npu.mem_allocate(1024, 0)?;
    let tasks_mem = npu.mem_allocate(1024, RKNPU_MEM_KERNEL_MAPPING)?;
    let mut input_mem = npu.mem_allocate(4096, 0)?;
    let mut weights_mem = npu.mem_allocate(4096, 0)?;
    let mut output_mem = npu.mem_allocate(4096, 0)?;

    println!(
        "Memory allocated: input_dma=0x{:x}, output_dma=0x{:x}, weights_dma=0x{:x}",
        input_mem.dma_addr(),
        output_mem.dma_addr(),
        weights_mem.dma_addr()
    );

    // Reset NPU
    npu.reset()?;

    // Generate matrix multiplication task
    let mut npu_regs = [0u64; 112];
    let mut params = MatmulParams {
        m: M as u16,
        k: 64, // Padded K dimension
        n: N as u16,
        input_dma: input_mem.dma_addr() as u32,
        weights_dma: weights_mem.dma_addr() as u32,
        output_dma: output_mem.dma_addr() as u32,
        tasks: npu_regs.as_mut_ptr(),
        fp32tofp16: 0, // Output as FP32
    };

    gen_matmul_fp16(&mut params).map_err(|e| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("gen_matmul_fp16 failed: {}", e),
        )
    })?;

    // Copy register commands to memory
    let regcmd_slice = regcmd_mem.as_slice_mut();
    unsafe {
        std::ptr::copy_nonoverlapping(
            npu_regs.as_ptr() as *const u8,
            regcmd_slice.as_mut_ptr(),
            std::mem::size_of_val(&npu_regs),
        );
    }

    // Initialize memory
    let input_slice = input_mem.as_slice_mut();
    let weights_slice = weights_mem.as_slice_mut();
    let output_slice = output_mem.as_slice_mut();
    
    input_slice[..M * 64 * 2].fill(0);
    weights_slice[..64 * N * 2].fill(0);
    output_slice[..M * N * 4].fill(0);

    // Setup task structure
    let tasks_ptr = tasks_mem.as_ptr();
    let tasks = unsafe { &mut *(tasks_ptr as *mut RknpuTask) };
    
    tasks.flags = 0;
    tasks.op_idx = 0;
    tasks.enable_mask = 0xd;
    tasks.int_mask = 0x300; // Wait for DPU to finish
    tasks.int_clear = 0x1ffff;
    tasks.int_status = 0;
    tasks.regcfg_amount = (npu_regs.len() as u32) - (RKNPU_PC_DATA_EXTRA_AMOUNT + 4);
    tasks.regcfg_offset = 0;
    tasks.regcmd_addr = regcmd_mem.dma_addr();

    // Fill weights (Matrix B) in FP16 format
    let weights_ptr = weights_mem.as_ptr();
    let weights_fp16 = unsafe { std::slice::from_raw_parts_mut(weights_ptr as *mut f16, 64 * N) };
    for n in 1..=N {
        for k in 1..=K {
            let idx = weight_fp16(64, n as i32, k as i32) as usize;
            let value = MATRIX_B[(n - 1) * K + (k - 1)];
            weights_fp16[idx] = f16::from_f32(value);
        }
    }

    // Fill input (Matrix A) in FP16 format
    let input_ptr = input_mem.as_ptr();
    let input_fp16 = unsafe { std::slice::from_raw_parts_mut(input_ptr as *mut f16, M * 64) };
    for m in 1..=M {
        for k in 1..=K {
            let idx = feature_data(64, 4, 1, 8, k as i32, m as i32, 1) as usize;
            let value = MATRIX_A[(m - 1) * K + (k - 1)];
            input_fp16[idx] = f16::from_f32(value);
        }
    }

    // Submit task to NPU
    let mut submit = RknpuSubmit {
        flags: RKNPU_JOB_PC | RKNPU_JOB_BLOCK | RKNPU_JOB_PINGPONG,
        timeout: 6000,
        task_start: 0,
        task_number: 1,
        task_counter: 0,
        priority: 0,
        task_obj_addr: tasks_mem.obj_addr(),
        regcfg_obj_addr: 0,
        task_base_addr: 0,
        user_data: 0,
        core_mask: 1,
        fence_fd: -1,
        subcore_task: [
            RknpuSubcoreTask {
                task_start: 0,
                task_number: 1,
            },
            RknpuSubcoreTask {
                task_start: 1,
                task_number: 0,
            },
            RknpuSubcoreTask {
                task_start: 2,
                task_number: 0,
            },
            RknpuSubcoreTask {
                task_start: 0,
                task_number: 0,
            },
            RknpuSubcoreTask {
                task_start: 0,
                task_number: 0,
            },
        ],
    };

    npu.submit(&mut submit)?;
    println!("Task submitted successfully");

    // Verify results
    println!(
        "========================================================================================================="
    );
    let output_ptr = output_mem.as_ptr();
    let output_data = unsafe { std::slice::from_raw_parts(output_ptr as *const f32, M * N) };
    let mut all_match = true;

    for m in 1..=M {
        for n in 1..(N) {
            let idx = feature_data(N as i32, 4, 1, 4, n as i32, m as i32, 1) as usize;
            let actual = output_data[idx];
            let expected = EXPECTED_RESULT[(m - 1) * N + (n - 1)];
            if (actual - expected).abs() > 0.1 {
                println!(
                    "mismatch m:{}  n:{}  expected:{:6.1}  actual:{:6.1}",
                    m, n, expected, actual
                );
                all_match = false;
            }
        }
        println!();
    }
    println!(
        "========================================================================================================="
    );

    // Cleanup is automatic when NpuMemory objects are dropped

    if all_match {
        println!("\n✓ All results match expected values!");
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::Other,
            "Results do not match expected values",
        ))
    }
}
