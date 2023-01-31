use std::time::Duration;

pub mod consumer;
pub mod loadgen;
pub mod producer;
pub mod utils;

// Time to execute a single inference in ms
pub const SINGLE_REQUEST_DELAY: u128 = 1;

// Reconfiguration timeout in seconds
pub const RECONFIG_TIMEOUT: Duration = Duration::from_secs(10);
