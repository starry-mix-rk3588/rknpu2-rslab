use crate::RknnError;
use crate::RknnResult;
use crate::rknn_tensor_attr;
use core::panic;
use ndarray::prelude::*;

use std::ffi::c_void;
use std::mem;
use std::ptr;

pub type RKNNContext = u64;

#[derive(Debug)]
pub struct RKNNContextPack {
    pub ctx: RKNNContext,
    pub io_info: RKNNInputOutputNumber,
    pub input_info: Vec<rknn_tensor_attr>,
    pub output_info: Vec<rknn_tensor_attr>,
    pub input_shape: Vec<u32>,
    pub is_quant: bool,
}

/// Make rknn context with useful informations.  
/// Note: This function now only support single input
pub fn make_rknn_context_pack(ctx: RKNNContext) -> RknnResult<RKNNContextPack> {
    let io_info = get_input_output_number(ctx)?;
    let input_info = get_model_input_info(ctx, io_info.n_input)?;
    let output_info = get_model_output_info(ctx, io_info.n_output)?;
    let input_info_0 = input_info.first().unwrap();
    let input_shape = Vec::from(&input_info_0.dims[0..input_info_0.n_dims as usize]);
    let qnt_type = input_info_0.qnt_type;
    let is_quant = match qnt_type {
        crate::_rknn_tensor_qnt_type_RKNN_TENSOR_QNT_NONE => false,
        crate::_rknn_tensor_qnt_type_RKNN_TENSOR_QNT_DFP => true,
        crate::_rknn_tensor_qnt_type_RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC => true,
        crate::_rknn_tensor_qnt_type_RKNN_TENSOR_QNT_MAX => true,
        _ => {
            panic!("Error, Unknown quant type.")
        }
    };
    let pack = RKNNContextPack {
        ctx,
        io_info,
        input_info,
        output_info,
        input_shape,
        is_quant,
    };

    Ok(pack)
}

pub type RKNNInitExtend = crate::rknn_init_extend;
pub fn rknn_init(
    model: Vec<u8>,
    flag: u32,
    rknn_init_extend: Option<&mut RKNNInitExtend>,
) -> RknnResult<RKNNContext> {
    let mut ctx: RKNNContext = 0;
    let ctx = ptr::from_mut(&mut ctx);
    let model_len = model.len() as u32;
    let mut model = model;
    let model: *mut c_void = model.as_mut_ptr() as *mut c_void;

    let rknn_init_extend = match rknn_init_extend {
        Some(rknn_init_extend) => ptr::from_mut(rknn_init_extend),
        None => ptr::null_mut(),
    };

    unsafe {
        let ret = crate::rknn_init(ctx, model, model_len, flag, rknn_init_extend);
        if ret == 0 {
            Ok(*ctx)
        } else {
            Err(RknnError::from_code(ret))
        }
    }
}

pub struct RKNNSdkVersion {
    pub api_version: String,
    pub drv_version: String,
}
pub fn get_sdk_version(ctx: RKNNContext) -> RknnResult<RKNNSdkVersion> {
    let mut version: crate::rknn_sdk_version = crate::rknn_sdk_version {
        api_version: [0; 256],
        drv_version: [0; 256],
    };
    let version_ptr = &mut version as *mut _ as *mut c_void;
    let size = mem::size_of::<crate::rknn_sdk_version>() as u32;

    unsafe {
        let ret = crate::rknn_query(
            ctx,
            crate::_rknn_query_cmd_RKNN_QUERY_SDK_VERSION,
            version_ptr,
            size,
        );
        if ret == 0 {
            let api_version_cstr = std::ffi::CStr::from_ptr(version.api_version.as_ptr());
            let drv_version_cstr = std::ffi::CStr::from_ptr(version.drv_version.as_ptr());
            let api_version = api_version_cstr.to_str().unwrap().to_string();
            let drv_version = drv_version_cstr.to_str().unwrap().to_string();
            Ok(RKNNSdkVersion {
                api_version,
                drv_version,
            })
        } else {
            Err(RknnError::from_code(ret))
        }
    }
}

pub type RKNNInputOutputNumber = crate::rknn_input_output_num;

pub fn get_input_output_number(ctx: RKNNContext) -> RknnResult<RKNNInputOutputNumber> {
    let cmd = crate::_rknn_query_cmd_RKNN_QUERY_IN_OUT_NUM;
    let mut input: crate::rknn_input_output_num = RKNNInputOutputNumber {
        n_input: 0,
        n_output: 0,
    };
    let size = mem::size_of::<crate::rknn_input_output_num>() as u32;
    let input_ptr = &mut input as *mut _ as *mut c_void;

    unsafe {
        let ret = crate::rknn_query(ctx, cmd, input_ptr, size);
        if ret == 0 {
            Ok(input)
        } else {
            Err(RknnError::from_code(ret))
        }
    }
}

pub type RKNNTensorAttr = crate::rknn_tensor_attr;
pub fn get_model_input_info(ctx: RKNNContext, input_num: u32) -> RknnResult<Vec<RKNNTensorAttr>> {
    let mut input_attrs: Vec<RKNNTensorAttr> = Vec::with_capacity(input_num as usize);

    // metset(input, 0, sizeof(input));
    unsafe {
        for _ in 0..input_attrs.capacity() {
            input_attrs.push(mem::zeroed());
        }
    }

    let cmd = crate::_rknn_query_cmd_RKNN_QUERY_INPUT_ATTR;
    let size = mem::size_of::<RKNNTensorAttr>() as u32;
    for i in 0..input_num as usize {
        let entry = input_attrs.get_mut(i).unwrap();
        entry.index = i as u32;
        let entry_ptr = entry as *mut _ as *mut c_void;
        unsafe {
            let ret = crate::rknn_query(ctx, cmd, entry_ptr, size);
            if ret != 0 {
                return Err(RknnError::from_code(ret));
            }
        }
    }
    Ok(input_attrs)
}

pub fn get_model_output_info(ctx: RKNNContext, output_num: u32) -> RknnResult<Vec<RKNNTensorAttr>> {
    let output_num = output_num as usize;
    // rknn_tensor_attr output_attrs[io_num.n_output];
    let mut output_attrs: Vec<RKNNTensorAttr> = Vec::with_capacity(output_num);

    // memset(output_attrs, 0, sizeof(output_attrs));
    unsafe {
        for _ in 0..output_attrs.capacity() {
            output_attrs.push(mem::zeroed());
        }
    }

    let cmd = crate::_rknn_query_cmd_RKNN_QUERY_OUTPUT_ATTR;
    let size = mem::size_of::<RKNNTensorAttr>() as u32;
    for i in 0..output_num {
        let entry = output_attrs.get_mut(i).unwrap();
        let entry_ptr = entry as *mut _ as *mut c_void;
        entry.index = i as u32;
        unsafe {
            let ret = crate::rknn_query(ctx, cmd, entry_ptr, size);
            if ret != 0 {
                return Err(RknnError::from_code(ret));
            }
        }
    }

    Ok(output_attrs)
}

pub type RKNNInput = crate::rknn_input;
pub type RKNNOutput = crate::rknn_output;
pub fn make_rknn_image_input(mut image_array_view: ArrayViewMut<u8, IxDyn>) -> Vec<RKNNInput> {
    // RKNNInput tensor_inputs;
    let tensor_inputs_layout = std::alloc::Layout::new::<RKNNInput>();
    let mut tensor_inputs: RKNNInput =
        unsafe { *(std::alloc::alloc_zeroed(tensor_inputs_layout) as *mut RKNNInput) };

    tensor_inputs.index = 0;
    tensor_inputs.type_ = crate::_rknn_tensor_type_RKNN_TENSOR_UINT8;
    tensor_inputs.fmt = crate::_rknn_tensor_format_RKNN_TENSOR_NHWC;
    tensor_inputs.size = image_array_view.len() as u32;
    tensor_inputs.buf = image_array_view.as_mut_ptr() as *mut _ as *mut c_void;

    vec![tensor_inputs]
}

pub fn rknn_inputs_set(
    ctx: RKNNContext,
    input_num: u32,
    mut inputs: Vec<RKNNInput>,
) -> Result<i32, i32> {
    let inputs_ptr = inputs.as_mut_ptr() as *mut RKNNInput;
    unsafe {
        let ret = crate::rknn_inputs_set(ctx, input_num, inputs_ptr);
        if ret == 0 { Ok(ret) } else { Err(ret) }
    }
}

pub fn rknn_run(ctx: RKNNContext) -> Result<i32, i32> {
    unsafe {
        let ret = crate::rknn_run(ctx, ptr::null_mut());
        if ret == 0 { Ok(ret) } else { Err(ret) }
    }
}

pub fn rknn_outputs_get(ctx: RKNNContext, output_num: u32) -> Result<Vec<RKNNOutput>, i32> {
    let mut outputs: Vec<RKNNOutput> = Vec::with_capacity(output_num as usize);
    // memset(outputs, 0, sizeof(outputs));
    unsafe {
        for _ in 0..outputs.capacity() {
            outputs.push(mem::zeroed());
        }
    }
    outputs[0].want_float = 1; // get float data
    let outputs_ptr = outputs.as_mut_ptr();
    unsafe {
        let ret = crate::rknn_outputs_get(ctx, output_num, outputs_ptr, ptr::null_mut());
        if ret == 0 { Ok(outputs) } else { Err(ret) }
    }
}
