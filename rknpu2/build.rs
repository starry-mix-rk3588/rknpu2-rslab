use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let lib_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .join("rknpu2-sys")
        .join("library");

    // 告诉 Cargo 如果指定的库目录变化了，重新运行构建脚本
    println!("cargo:rerun-if-changed={}", lib_dir.display());

    // 设置链接器搜索路径（编译时）
    println!("cargo:rustc-link-search={}", lib_dir.display());

    // 设置 RPATH (运行时搜索路径)
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
}
