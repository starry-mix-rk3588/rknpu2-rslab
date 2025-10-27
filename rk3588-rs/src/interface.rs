//! Device and memory interface for RK3588 NPU
//! This module requires std feature (nix dependency)

use crate::ioctl::*;
use nix::sys::mman::{mmap, munmap, MapFlags, ProtFlags};
use std::cell::RefCell;
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io;
use std::num::NonZeroUsize;
use std::os::unix::io::{AsRawFd, RawFd};
use std::ptr::NonNull;

/// NPU device handle
pub struct NpuDevice {
    file: File,
    allocated_resources: RefCell<HashSet<(u32, u64)>>,
}

impl NpuDevice {
    /// Open the NPU device
    pub fn open() -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/dri/card1")?;

        let fd = file.as_raw_fd();

        // Query DRM version
        let mut name_buf = vec![0u8; 256];
        let mut date_buf = vec![0u8; 256];
        let mut desc_buf = vec![0u8; 256];

        let mut drm_ver = DrmVersion {
            version_major: 0,
            version_minor: 0,
            version_patchlevel: 0,
            name_len: name_buf.len(),
            name: name_buf.as_mut_ptr(),
            date_len: date_buf.len(),
            date: date_buf.as_mut_ptr(),
            desc_len: desc_buf.len(),
            desc: desc_buf.as_mut_ptr(),
        };

        unsafe {
            drm_ioctl_version(fd, &mut drm_ver).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("DRM_IOCTL_VERSION failed: {}", e),
                )
            })?;
        }

        // Convert buffers to strings
        let name = String::from_utf8_lossy(&name_buf[..drm_ver.name_len]);
        let date = String::from_utf8_lossy(&date_buf[..drm_ver.date_len]);
        let desc = String::from_utf8_lossy(&desc_buf[..drm_ver.desc_len]);

        println!("drm name is {} - {} - {}", name, date, desc);

        Ok(NpuDevice {
            file,
            allocated_resources: RefCell::new(HashSet::new()),
        })
    }

    /// Get the raw file descriptor
    pub fn as_raw_fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }

    /// Reset the NPU
    pub fn reset(&self) -> io::Result<()> {
        let mut action = RknpuActionStruct {
            flags: 6, // reset 6 
            value: 0,
        };

        unsafe {
            drm_ioctl_rknpu_action(self.as_raw_fd(), &mut action).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("RKNPU_ACT_RESET failed: {}", e),
                )
            })?;
        }

        Ok(())
    }

    /// Allocate memory
    pub fn mem_allocate(&self, size: usize, flags: u32) -> io::Result<NpuMemory> {
        let mut mem_create = RknpuMemCreate {
            handle: 0,
            flags: flags | RKNPU_MEM_NON_CACHEABLE,
            size: size as u64,
            obj_addr: 0,
            dma_addr: 0,
            sram_size: 0,
        };

        unsafe {
            drm_ioctl_rknpu_mem_create(self.as_raw_fd(), &mut mem_create).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("RKNPU_MEM_CREATE failed: {}", e),
                )
            })?;
        }

        println!("===================== RknpuMemCreate :{:#?}", mem_create);

        let mut mem_map = RknpuMemMap {
            handle: mem_create.handle,
            reserved: 0,
            offset: 0,
        };

        unsafe {
            drm_ioctl_rknpu_mem_map(self.as_raw_fd(), &mut mem_map).map_err(|e| {
                io::Error::new(io::ErrorKind::Other, format!("RKNPU_MEM_MAP failed: {}", e))
            })?;
        }

        println!("====================== RknpuMemMap: {:#?}", mem_map);

        let map_ptr = unsafe {
            mmap(
                None,
                NonZeroUsize::new(size).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidInput, "Size cannot be zero")
                })?,
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_SHARED,
                &self.file,
                mem_map.offset as i64,
            )
            .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("mmap failed: {}", e)))?
        };

        // Record the allocated resource
        self.allocated_resources
            .borrow_mut()
            .insert((mem_create.handle, mem_create.obj_addr));

        Ok(NpuMemory {
            ptr: map_ptr.as_ptr().cast::<u8>(),
            size,
            handle: mem_create.handle,
            obj_addr: mem_create.obj_addr,
            dma_addr: mem_create.dma_addr,
        })
    }

    /// Destroy allocated memory
    pub fn mem_destroy(&self, handle: u32, obj_addr: u64) -> io::Result<()> {
        let mut destroy = RknpuMemDestroy {
            handle,
            reserved: 0,
            obj_addr,
        };

        unsafe {
            drm_ioctl_rknpu_mem_destroy(self.as_raw_fd(), &mut destroy).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::Other,
                    format!("RKNPU_MEM_DESTROY failed: {}", e),
                )
            })?;
        }

        // Remove the resource from tracking
        self.allocated_resources.borrow_mut().remove(&(handle, obj_addr));

        Ok(())
    }

    /// Submit a task to the NPU
    pub fn submit(&self, submit: &mut RknpuSubmit) -> io::Result<()> {
        unsafe {
            drm_ioctl_rknpu_submit(self.as_raw_fd(), submit).map_err(|e| {
                io::Error::new(io::ErrorKind::Other, format!("RKNPU_SUBMIT failed: {}", e))
            })?;
        }

        Ok(())
    }
}

impl Drop for NpuDevice {
    fn drop(&mut self) {
        // Clean up all remaining allocated resources
        let resources = self.allocated_resources.borrow().clone();
        for (handle, obj_addr) in resources {
            let mut destroy = RknpuMemDestroy {
                handle,
                reserved: 0,
                obj_addr,
            };
            
            unsafe {
                let _ = drm_ioctl_rknpu_mem_destroy(self.as_raw_fd(), &mut destroy);
            }
        }
        
        // File will be automatically closed when dropped
    }
}

/// Represents allocated NPU memory
pub struct NpuMemory {
    ptr: *mut u8,
    size: usize,
    handle: u32,
    obj_addr: u64,
    dma_addr: u64,
}

impl NpuMemory {
    /// Get the mapped pointer
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Get the size of the allocation
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the memory handle
    pub fn handle(&self) -> u32 {
        self.handle
    }

    /// Get the object address
    pub fn obj_addr(&self) -> u64 {
        self.obj_addr
    }

    /// Get the DMA address
    pub fn dma_addr(&self) -> u64 {
        self.dma_addr
    }

    /// Get a mutable slice view of the memory
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Get an immutable slice view of the memory
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// Write data to the memory
    pub fn write(&mut self, data: &[u8]) -> io::Result<()> {
        if data.len() > self.size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Data size exceeds allocated memory size",
            ));
        }

        let slice = self.as_slice_mut();
        slice[..data.len()].copy_from_slice(data);
        Ok(())
    }
}

impl Drop for NpuMemory {
    fn drop(&mut self) {
        // munmap is automatically called when the pointer goes out of scope
        if let Some(ptr) = NonNull::new(self.ptr.cast()) {
            if let Some(size) = NonZeroUsize::new(self.size) {
                unsafe {
                    let _ = munmap(ptr, size.get());
                }
            }
        }
    }
}
