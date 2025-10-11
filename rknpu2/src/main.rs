mod matmul;

use matmul::run_matmul_test;

fn main() {
    println!("=== RK3588 NPU Matrix Multiplication Test ===\n");
    
    match run_matmul_test() {
        Ok(()) => {
            println!("\n✓ Matrix multiplication test completed successfully!");
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("\n✗ Matrix multiplication test failed: {}", e);
            std::process::exit(1);
        }
    }
}
