use std::hint::spin_loop;

pub fn delay_loop(delay_in_ms: u128) {
    let start = std::time::Instant::now();
    while start.elapsed().as_millis() < delay_in_ms {
        spin_loop();
    }
}

pub fn lower_power_of_two(cores: i64) -> i64 {
    2_i64.pow((cores as f64).log2().floor() as u32)
}
