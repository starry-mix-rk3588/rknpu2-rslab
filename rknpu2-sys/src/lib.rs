#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unused_mut)]
#![allow(unused)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub mod rknn;

const RKNN_SUCCESS: i32 = 0;

/// RKNN result type for better error handling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RknnError {
    /// Execute succeed
    Success,
    /// Execute failed
    Fail,
    /// Execute timeout
    Timeout,
    /// Device is unavailable
    DeviceUnavailable,
    /// Memory malloc fail
    MallocFail,
    /// Parameter is invalid
    ParamInvalid,
    /// Model is invalid
    ModelInvalid,
    /// Context is invalid
    CtxInvalid,
    /// Input is invalid
    InputInvalid,
    /// Output is invalid
    OutputInvalid,
    /// The device is unmatch, please update rknn sdk and npu driver/firmware
    DeviceUnmatch,
    /// This RKNN model use pre_compile mode, but not compatible with current driver
    IncompatiblePreCompileModel,
    /// This RKNN model set optimization level, but not compatible with current driver
    IncompatibleOptimizationLevelVersion,
    /// This RKNN model set target platform, but not compatible with current platform
    TargetPlatformUnmatch,
    /// Unknown error code
    Unknown(i32),
}

impl RknnError {
    /// Convert from raw error code to RknnError enum
    pub fn from_code(code: i32) -> Self {
        match code {
            RKNN_SUCCESS => RknnError::Success,
            RKNN_ERR_FAIL => RknnError::Fail,
            RKNN_ERR_TIMEOUT => RknnError::Timeout,
            RKNN_ERR_DEVICE_UNAVAILABLE => RknnError::DeviceUnavailable,
            RKNN_ERR_MALLOC_FAIL => RknnError::MallocFail,
            RKNN_ERR_PARAM_INVALID => RknnError::ParamInvalid,
            RKNN_ERR_MODEL_INVALID => RknnError::ModelInvalid,
            RKNN_ERR_CTX_INVALID => RknnError::CtxInvalid,
            RKNN_ERR_INPUT_INVALID => RknnError::InputInvalid,
            RKNN_ERR_OUTPUT_INVALID => RknnError::OutputInvalid,
            RKNN_ERR_DEVICE_UNMATCH => RknnError::DeviceUnmatch,
            RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL => RknnError::IncompatiblePreCompileModel,
            RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION => RknnError::IncompatibleOptimizationLevelVersion,
            RKNN_ERR_TARGET_PLATFORM_UNMATCH => RknnError::TargetPlatformUnmatch,
            _ => RknnError::Unknown(code),
        }
    }

    /// Convert to raw error code
    pub fn to_code(&self) -> i32 {
        match self {
            RknnError::Success => RKNN_SUCCESS,
            RknnError::Fail => RKNN_ERR_FAIL,
            RknnError::Timeout => RKNN_ERR_TIMEOUT,
            RknnError::DeviceUnavailable => RKNN_ERR_DEVICE_UNAVAILABLE,
            RknnError::MallocFail => RKNN_ERR_MALLOC_FAIL,
            RknnError::ParamInvalid => RKNN_ERR_PARAM_INVALID,
            RknnError::ModelInvalid => RKNN_ERR_MODEL_INVALID,
            RknnError::CtxInvalid => RKNN_ERR_CTX_INVALID,
            RknnError::InputInvalid => RKNN_ERR_INPUT_INVALID,
            RknnError::OutputInvalid => RKNN_ERR_OUTPUT_INVALID,
            RknnError::DeviceUnmatch => RKNN_ERR_DEVICE_UNMATCH,
            RknnError::IncompatiblePreCompileModel => RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL,
            RknnError::IncompatibleOptimizationLevelVersion => RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION,
            RknnError::TargetPlatformUnmatch => RKNN_ERR_TARGET_PLATFORM_UNMATCH,
            RknnError::Unknown(code) => *code,
        }
    }

    /// Check if the error represents success
    pub fn is_success(&self) -> bool {
        matches!(self, RknnError::Success)
    }

    /// Check if the error represents a failure
    pub fn is_error(&self) -> bool {
        !self.is_success()
    }
}

impl std::fmt::Display for RknnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RknnError::Success => write!(f, "Execute succeed"),
            RknnError::Fail => write!(f, "Execute failed"),
            RknnError::Timeout => write!(f, "Execute timeout"),
            RknnError::DeviceUnavailable => write!(f, "Device is unavailable"),
            RknnError::MallocFail => write!(f, "Memory malloc fail"),
            RknnError::ParamInvalid => write!(f, "Parameter is invalid"),
            RknnError::ModelInvalid => write!(f, "Model is invalid"),
            RknnError::CtxInvalid => write!(f, "Context is invalid"),
            RknnError::InputInvalid => write!(f, "Input is invalid"),
            RknnError::OutputInvalid => write!(f, "Output is invalid"),
            RknnError::DeviceUnmatch => write!(f, "The device is unmatch, please update rknn sdk and npu driver/firmware"),
            RknnError::IncompatiblePreCompileModel => write!(f, "This RKNN model use pre_compile mode, but not compatible with current driver"),
            RknnError::IncompatibleOptimizationLevelVersion => write!(f, "This RKNN model set optimization level, but not compatible with current driver"),
            RknnError::TargetPlatformUnmatch => write!(f, "This RKNN model set target platform, but not compatible with current platform"),
            RknnError::Unknown(code) => write!(f, "Unknown error code: {}", code),
        }
    }
}

impl std::error::Error for RknnError {}

/// Result type for RKNN operations
pub type RknnResult<T> = Result<T, RknnError>;
