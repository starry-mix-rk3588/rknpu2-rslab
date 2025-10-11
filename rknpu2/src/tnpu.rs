use image::ImageBuffer;
use itertools::Itertools;
use ndarray::{ArrayViewMut, IxDyn};
use rknpu2_sys::rknn::{self, RKNNTensorAttr};
use std::time::Instant;

const MODEL_MOBILENET_V1: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/model/mobilenet_v1.rknn"
));
const IMAGE_DATA: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/model/iaa_pub9952.jpg"
));
const LABELS_TXT: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/model/labels.txt"));

fn image_to_array_view(
    img_buffer: &'_ mut ImageBuffer<image::Rgb<u8>, Vec<u8>>,
) -> ArrayViewMut<'_, u8, IxDyn> {
    let (w, h) = (img_buffer.width(), img_buffer.height());
    let arr = unsafe {
        let arr =
            ArrayViewMut::from_shape_ptr((w as usize, h as usize, 3), img_buffer.as_mut_ptr());
        arr.into_dyn()
    };

    arr
}

fn dump_attr_info(attr: RKNNTensorAttr) {
    println!("index: {}", attr.index);
    let binding = attr
        .name
        .iter()
        .take_while(|&c| *c != 0)
        .cloned()
        .collect::<Vec<u8>>();
    let name = String::from_utf8_lossy(&binding);
    println!("name: {:?}", name);
    println!("n_dims: {}", attr.n_dims);
    println!("dims: {:?}", &attr.dims[0..attr.n_dims as usize]);
    println!("n_elems: {}", attr.n_elems);
    println!("size: {}", attr.size);
    println!("fmt: {}", attr.fmt);
    println!("type_: {}", attr.type_);
    println!("qnt_type: {}", attr.qnt_type);
}

fn ptr_to_u8_slice<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

pub fn tnpu_main() {
    println!("RKNPU2 Rust Example");
    // Load Model
    let model = MODEL_MOBILENET_V1.to_vec();

    // read image and convert image to ArrayView
    let mut img = image::load_from_memory(IMAGE_DATA).unwrap().to_rgb8();
    let img_arr = image_to_array_view(&mut img);

    let ctx = rknn::rknn_init(model, 0, None).unwrap();

    // Get SDK Version
    let sdk_info = rknn::get_sdk_version(ctx).unwrap();
    println!(
        "api_version: {}, drv_version: {}",
        sdk_info.api_version, sdk_info.drv_version
    );

    let io_info = rknn::get_input_output_number(ctx).unwrap();
    println!(
        "n_input: {}, n_output: {}",
        io_info.n_input, io_info.n_output
    );

    let input_info = rknn::get_model_input_info(ctx, io_info.n_input).unwrap();
    dump_attr_info(input_info[0]);
    let output_info = rknn::get_model_output_info(ctx, io_info.n_output).unwrap();
    dump_attr_info(output_info[0]);

    // Get Model Pack
    let ctx_pack = rknn::make_rknn_context_pack(ctx).unwrap();

    // setup rknn input
    let rknn_inputs = rknn::make_rknn_image_input(img_arr);
    let buffer = ptr_to_u8_slice(rknn_inputs[0].buf as *const u8, 50);
    println!("rknn_inputs: {:?}, buffer: {:x?}", rknn_inputs, buffer);
    let _ret = rknn::rknn_inputs_set(ctx, io_info.n_input, rknn_inputs);

    // run rknn
    let start = Instant::now();
    let _ret = rknn::rknn_run(ctx).unwrap();

    // extract rknn outputs
    let rknn_outputs = rknn::rknn_outputs_get(ctx, io_info.n_output).unwrap();
    dbg!(start.elapsed());
    let dims = &ctx_pack.output_info[0].dims[0..ctx_pack.output_info[0].n_dims as usize];
    println!("output_info: {:?}", dims);
    println!(
        "rknn_outputs: {:?}, n_output: {}",
        rknn_outputs, io_info.n_output
    );
    let output = &rknn_outputs[0];
    let buffer_ptr = output.buf as *const u8;
    let floats =
        unsafe { std::slice::from_raw_parts(buffer_ptr as *const f32, (output.size / 4) as usize) };
    let logits = floats.to_vec();

    // Top 5
    let top5 = logits
        .into_iter()
        .enumerate()
        .sorted_by(|&(_, a), &(_, b)| b.partial_cmp(&a).unwrap())
        .take(5)
        .collect::<Vec<_>>();

    let labels = String::from_utf8_lossy(LABELS_TXT);
    let labels: Vec<&str> = labels.lines().collect();
    println!("Top 5:");
    for (i, v) in top5 {
        println!("  {}: {} ({}%)", i, labels[i], v * 100.0);
    }
}
