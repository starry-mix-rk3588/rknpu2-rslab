use std::{ffi::c_void, fs, ptr};
pub type RKNNContext = u64;

fn main() {
    let mut model = fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/model/mobilenet_v1.rknn")).unwrap();
    let mut ctx: RKNNContext = 0;
    let ctx = ptr::from_mut(&mut ctx);
    let model_len = model.len() as u32;
    let model: *mut c_void = model.as_mut_ptr() as *mut c_void;
    println!("Model Info: {}", model_len);
    for i in 0..16 {
        print!("{:02X} ", unsafe { *(model as *const u8).add(i) });
    }
    println!();

    unsafe {
        let res = rknpu2_sys::rknn_init(ctx, model, model_len, 0, ptr::null_mut());
        println!("rknn_init result: {}", res);
    }
}
