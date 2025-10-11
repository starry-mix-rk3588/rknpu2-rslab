#![cfg_attr(feature = "no_std", no_std)]

pub mod cna;
pub mod dpu;
pub mod hw;
pub mod ioctl;
pub mod matmul;

// Only include interface module when std feature is enabled (nix dependency)
#[cfg(feature = "std")]
pub mod interface;

pub use cna::*;
pub use dpu::*;
pub use hw::*;
pub use ioctl::*;
pub use matmul::*;

#[cfg(feature = "std")]
pub use interface::*;
