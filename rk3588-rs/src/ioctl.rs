#[cfg(not(feature = "no_std"))]
use std::os::raw::c_int;
#[cfg(feature = "no_std")]
use core::ffi::c_int;

// Register offsets
pub const RKNPU_OFFSET_VERSION: u32 = 0x0;
pub const RKNPU_OFFSET_VERSION_NUM: u32 = 0x4;
pub const RKNPU_OFFSET_PC_OP_EN: u32 = 0x8;
pub const RKNPU_OFFSET_PC_DATA_ADDR: u32 = 0x10;
pub const RKNPU_OFFSET_PC_DATA_AMOUNT: u32 = 0x14;
pub const RKNPU_OFFSET_PC_TASK_CONTROL: u32 = 0x30;
pub const RKNPU_OFFSET_PC_DMA_BASE_ADDR: u32 = 0x34;
pub const RKNPU_OFFSET_PC_TASK_STATUS: u32 = 0x3c;

pub const RKNPU_OFFSET_INT_MASK: u32 = 0x20;
pub const RKNPU_OFFSET_INT_CLEAR: u32 = 0x24;
pub const RKNPU_OFFSET_INT_STATUS: u32 = 0x28;
pub const RKNPU_OFFSET_INT_RAW_STATUS: u32 = 0x2c;

pub const RKNPU_OFFSET_CLR_ALL_RW_AMOUNT: u32 = 0x8010;
pub const RKNPU_OFFSET_DT_WR_AMOUNT: u32 = 0x8034;
pub const RKNPU_OFFSET_DT_RD_AMOUNT: u32 = 0x8038;
pub const RKNPU_OFFSET_WT_RD_AMOUNT: u32 = 0x803c;

pub const RKNPU_OFFSET_ENABLE_MASK: u32 = 0xf008;

pub const RKNPU_INT_CLEAR: u32 = 0x1ffff;

pub const RKNPU_PC_DATA_EXTRA_AMOUNT: u32 = 4;

// Memory type flags
pub const RKNPU_MEM_CONTIGUOUS: u32 = 0 << 0;
pub const RKNPU_MEM_NON_CONTIGUOUS: u32 = 1 << 0;
pub const RKNPU_MEM_NON_CACHEABLE: u32 = 0 << 1;
pub const RKNPU_MEM_CACHEABLE: u32 = 1 << 1;
pub const RKNPU_MEM_WRITE_COMBINE: u32 = 1 << 2;
pub const RKNPU_MEM_KERNEL_MAPPING: u32 = 1 << 3;
pub const RKNPU_MEM_IOMMU: u32 = 1 << 4;
pub const RKNPU_MEM_ZEROING: u32 = 1 << 5;
pub const RKNPU_MEM_SECURE: u32 = 1 << 6;
pub const RKNPU_MEM_NON_DMA32: u32 = 1 << 7;
pub const RKNPU_MEM_TRY_ALLOC_SRAM: u32 = 1 << 8;

pub const RKNPU_MEM_MASK: u32 = RKNPU_MEM_NON_CONTIGUOUS
    | RKNPU_MEM_CACHEABLE
    | RKNPU_MEM_WRITE_COMBINE
    | RKNPU_MEM_KERNEL_MAPPING
    | RKNPU_MEM_IOMMU
    | RKNPU_MEM_ZEROING
    | RKNPU_MEM_SECURE
    | RKNPU_MEM_NON_DMA32
    | RKNPU_MEM_TRY_ALLOC_SRAM;

// Sync mode flags
pub const RKNPU_MEM_SYNC_TO_DEVICE: u32 = 1 << 0;
pub const RKNPU_MEM_SYNC_FROM_DEVICE: u32 = 1 << 1;
pub const RKNPU_MEM_SYNC_MASK: u32 = RKNPU_MEM_SYNC_TO_DEVICE | RKNPU_MEM_SYNC_FROM_DEVICE;

// Job mode flags
pub const RKNPU_JOB_SLAVE: u32 = 0 << 0;
pub const RKNPU_JOB_PC: u32 = 1 << 0;
pub const RKNPU_JOB_BLOCK: u32 = 0 << 1;
pub const RKNPU_JOB_NONBLOCK: u32 = 1 << 1;
pub const RKNPU_JOB_PINGPONG: u32 = 1 << 2;
pub const RKNPU_JOB_FENCE_IN: u32 = 1 << 3;
pub const RKNPU_JOB_FENCE_OUT: u32 = 1 << 4;
pub const RKNPU_JOB_MASK: u32 = RKNPU_JOB_PC
    | RKNPU_JOB_NONBLOCK
    | RKNPU_JOB_PINGPONG
    | RKNPU_JOB_FENCE_IN
    | RKNPU_JOB_FENCE_OUT;

// Action flags
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RknpuAction {
    GetHwVersion = 0,
    GetDrvVersion = 1,
    GetFreq = 2,
    SetFreq = 3,
    GetVolt = 4,
    SetVolt = 5,
    ActReset = 6,
    GetBwPriority = 7,
    SetBwPriority = 8,
    GetBwExpect = 9,
    SetBwExpect = 10,
    GetBwTw = 11,
    SetBwTw = 12,
    ActClrTotalRwAmount = 13,
    GetDtWrAmount = 14,
    GetDtRdAmount = 15,
    GetWtRdAmount = 16,
    GetTotalRwAmount = 17,
    GetIommuEn = 18,
    SetProcNice = 19,
    PowerOn = 20,
    PowerOff = 21,
    GetTotalSramSize = 22,
    GetFreeSramSize = 23,
}

// User-desired buffer creation information structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RknpuMemCreate {
    pub handle: u32,
    pub flags: u32,
    pub size: u64,
    pub obj_addr: u64,
    pub dma_addr: u64,
    pub sram_size: u64,
}

// A structure for getting a fake-offset that can be used with mmap
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RknpuMemMap {
    pub handle: u32,
    pub reserved: u32,
    pub offset: u64,
}

// For destroying DMA buffer
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RknpuMemDestroy {
    pub handle: u32,
    pub reserved: u32,
    pub obj_addr: u64,
}

// For synchronizing DMA buffer
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RknpuMemSync {
    pub flags: u32,
    pub reserved: u32,
    pub obj_addr: u64,
    pub offset: u64,
    pub size: u64,
}

// struct rknpu_task structure for task information
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct RknpuTask {
    pub flags: u32,
    pub op_idx: u32,
    pub enable_mask: u32,
    pub int_mask: u32,
    pub int_clear: u32,
    pub int_status: u32,
    pub regcfg_amount: u32,
    pub regcfg_offset: u32,
    pub regcmd_addr: u64,
}

// struct rknpu_subcore_task structure for subcore task index
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RknpuSubcoreTask {
    pub task_start: u32,
    pub task_number: u32,
}

// struct rknpu_submit structure for job submit
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RknpuSubmit {
    pub flags: u32,
    pub timeout: u32,
    pub task_start: u32,
    pub task_number: u32,
    pub task_counter: u32,
    pub priority: i32,
    pub task_obj_addr: u64,
    pub regcfg_obj_addr: u64,
    pub task_base_addr: u64,
    pub user_data: u64,
    pub core_mask: u32,
    pub fence_fd: i32,
    pub subcore_task: [RknpuSubcoreTask; 5],
}

// struct rknpu_action structure for action (GET, SET or ACT)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RknpuActionStruct {
    pub flags: u32,
    pub value: u32,
}

// IOCTL command numbers
pub const RKNPU_ACTION: u32 = 0x00;
pub const RKNPU_SUBMIT: u32 = 0x01;
pub const RKNPU_MEM_CREATE: u32 = 0x02;
pub const RKNPU_MEM_MAP: u32 = 0x03;
pub const RKNPU_MEM_DESTROY: u32 = 0x04;
pub const RKNPU_MEM_SYNC: u32 = 0x05;

// DRM command base (from libdrm/drm.h)
pub const DRM_COMMAND_BASE: u32 = 0x40;

// IOCTL magic for DRM
pub const DRM_IOCTL_BASE: u8 = b'd';

// Helper to construct IOCTL numbers
// For DRM ioctls we use the nix::ioctl_readwrite! macro
#[cfg(feature = "std")]
nix::ioctl_readwrite!(
    drm_ioctl_rknpu_action,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + RKNPU_ACTION,
    RknpuActionStruct
);

#[cfg(feature = "std")]
nix::ioctl_readwrite!(
    drm_ioctl_rknpu_submit,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + RKNPU_SUBMIT,
    RknpuSubmit
);

#[cfg(feature = "std")]
nix::ioctl_readwrite!(
    drm_ioctl_rknpu_mem_create,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + RKNPU_MEM_CREATE,
    RknpuMemCreate
);

#[cfg(feature = "std")]
nix::ioctl_readwrite!(
    drm_ioctl_rknpu_mem_map,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + RKNPU_MEM_MAP,
    RknpuMemMap
);

#[cfg(feature = "std")]
nix::ioctl_readwrite!(
    drm_ioctl_rknpu_mem_destroy,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + RKNPU_MEM_DESTROY,
    RknpuMemDestroy
);

#[cfg(feature = "std")]
nix::ioctl_readwrite!(
    drm_ioctl_rknpu_mem_sync,
    DRM_IOCTL_BASE,
    DRM_COMMAND_BASE + RKNPU_MEM_SYNC,
    RknpuMemSync
);

// DRM version structure
#[repr(C)]
#[derive(Debug)]
pub struct DrmVersion {
    pub version_major: c_int,
    pub version_minor: c_int,
    pub version_patchlevel: c_int,
    pub name_len: usize,
    pub name: *mut u8,
    pub date_len: usize,
    pub date: *mut u8,
    pub desc_len: usize,
    pub desc: *mut u8,
}

// DRM_IOCTL_VERSION
#[cfg(feature = "std")]
nix::ioctl_readwrite!(drm_ioctl_version, DRM_IOCTL_BASE, 0x00, DrmVersion);
