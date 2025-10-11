use std::fs::File;
use std::io::Error;
use std::os::fd::AsRawFd;

use bstr::BString;
use uapi::{Pod, c, pod_zeroed};

pub unsafe fn ioctl<T>(fd: c::c_int, request: c::c_ulong, t: &mut T) -> Result<c::c_int, Error> {
    let mut ret;
    loop {
        ret = unsafe { c::ioctl(fd, request as c::c_ulong, &mut *t) };
        if ret != -1 {
            return Ok(ret);
        }
        let err = uapi::get_errno();
        if !matches!(err, c::EINTR | c::EAGAIN) {
            return Err(Error::from_raw_os_error(err));
        }
    }
}

pub const DRM_IOCTL_BASE: u64 = b'd' as u64;

pub const fn drm_io(nr: u64) -> u64 {
    uapi::_IO(DRM_IOCTL_BASE, nr)
}

pub const fn drm_iow<T>(nr: u64) -> u64 {
    uapi::_IOW::<T>(DRM_IOCTL_BASE, nr)
}

pub const fn drm_iowr<T>(nr: u64) -> u64 {
    uapi::_IOWR::<T>(DRM_IOCTL_BASE, nr)
}

#[repr(C)]
struct drm_version {
    version_major: c::c_int,
    version_minor: c::c_int,
    version_patchlevel: c::c_int,
    name_len: usize,
    name: *mut u8,
    date_len: usize,
    date: *mut u8,
    desc_len: usize,
    desc: *mut u8,
}

unsafe impl Pod for drm_version {}

const DRM_IOCTL_VERSION: u64 = drm_iowr::<drm_version>(0x00);

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DrmVersion {
    pub version_major: i32,
    pub version_minor: i32,
    pub version_patchlevel: i32,
    pub name: BString,
    pub date: BString,
    pub desc: BString,
}

pub fn get_version(fd: c::c_int) -> Result<DrmVersion, Error> {
    let mut name = Vec::<u8>::new();
    let mut date = Vec::<u8>::new();
    let mut desc = Vec::<u8>::new();
    let mut res: drm_version = pod_zeroed();
    loop {
        res.name_len = name.capacity();
        res.name = name.as_mut_ptr();
        res.date_len = date.capacity();
        res.date = date.as_mut_ptr();
        res.desc_len = desc.capacity();
        res.desc = desc.as_mut_ptr();
        unsafe {
            ioctl(fd, DRM_IOCTL_VERSION, &mut res)?;
        }
        if res.name_len <= name.capacity()
            && res.date_len <= date.capacity()
            && res.desc_len <= desc.capacity()
        {
            break;
        }
        name.reserve_exact(res.name_len);
        date.reserve_exact(res.date_len);
        desc.reserve_exact(res.desc_len);
    }
    unsafe {
        name.set_len(res.name_len);
        date.set_len(res.date_len);
        desc.set_len(res.desc_len);
    }
    Ok(DrmVersion {
        version_major: res.version_major,
        version_minor: res.version_minor,
        version_patchlevel: res.version_patchlevel,
        name: name.into(),
        date: date.into(),
        desc: desc.into(),
    })
}

// RKNPU specific ioctl definitions
// DRM_COMMAND_BASE is typically 0x40
const DRM_RKNPU_ACTION: u64 = 0x00;
const DRM_IOCTL_RKNPU_ACTION: u64 = drm_iowr::<RknpuAction>(0x40 + DRM_RKNPU_ACTION);

// RKNPU action flags
const RKNPU_ACTION_GET_HW_VERSION: u32 = 0x01;
const RKNPU_ACTION_GET_BWPRO: u32 = 0x02;
const RKNPU_ACTION_RESET: u32 = 0x03;

#[repr(C)]
struct RknpuAction {
    flags: u32,
    size: u32,
    data: u64,
}

unsafe impl Pod for RknpuAction {}

#[repr(C)]
struct RknpuHwVersion {
    version: u32,
}

unsafe impl Pod for RknpuHwVersion {}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct HardwareVersion {
    pub version: u32,
}

pub fn get_hw_version(fd: c::c_int) -> Result<HardwareVersion, Error> {
    let mut hw_ver: RknpuHwVersion = pod_zeroed();
    let mut action: RknpuAction = pod_zeroed();

    action.flags = RKNPU_ACTION_GET_HW_VERSION;
    action.size = std::mem::size_of::<RknpuHwVersion>() as u32;
    action.data = &mut hw_ver as *mut RknpuHwVersion as u64;

    unsafe {
        ioctl(fd, DRM_IOCTL_RKNPU_ACTION, &mut action)?;
    }

    Ok(HardwareVersion {
        version: hw_ver.version,
    })
}

pub fn scall_main() {
    println!("Hello, world!\n");
    let file = File::open("/dev/dri/card0").unwrap();
    let fd = file.as_raw_fd();

    let version = get_version(fd).unwrap();
    println!("DRM Version: {version:?}");

    println!("\n--- Getting Hardware Version ---");
    println!("DRM_IOCTL_RKNPU_ACTION: 0x{:08x}", DRM_IOCTL_RKNPU_ACTION);

    match get_hw_version(fd) {
        Ok(hw_ver) => {
            println!("✓ Hardware Version: 0x{:08x}", hw_ver.version);
            if hw_ver.version != 0 {
                println!(
                    "  NPU Version: {}.{}.{}",
                    (hw_ver.version >> 16) & 0xff,
                    (hw_ver.version >> 8) & 0xff,
                    hw_ver.version & 0xff
                );
            } else {
                println!("  Note: Hardware version is 0, which may indicate:");
                println!("  - The ioctl command number is incorrect");
                println!("  - The driver doesn't support this query");
                println!("  - Additional setup is required");
            }
        }
        Err(e) => {
            eprintln!("✗ Failed to get hardware version: {}", e);
            eprintln!("  Error code: {:?}", e.raw_os_error());
        }
    }
}
