use std::{
    fmt::{Display, Formatter},
    sync::mpsc::{Receiver, Sender},
};

use rand::prelude::Distribution;

use crate::RECONFIG_TIMEOUT;

pub trait LoadGen {
    fn run(&mut self, _sender: &mut Sender<u64>, _receiver: &mut Receiver<u64>) {}
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Constant;

impl Display for Constant {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Constant")
    }
}

impl Constant {
    fn tx_rx(&self, batch_size: usize, sender: &mut Sender<u64>, receiver: &mut Receiver<u64>) {
        // Send requests
        (0..batch_size).for_each(|_| while sender.send(1).is_err() {});

        // Wait for responses
        (0..batch_size).for_each(|_| while receiver.recv().is_err() {});
    }
}

impl LoadGen for Constant {
    fn run(&mut self, sender: &mut Sender<u64>, receiver: &mut Receiver<u64>) {
        let starting_bs = 16;
        let start = std::time::Instant::now();

        // Send requests for 100 seconds
        while start.elapsed().as_secs() < RECONFIG_TIMEOUT.as_secs() {
            self.tx_rx(starting_bs, sender, receiver);
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Diurnal;

impl Display for Diurnal {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Diurnal")
    }
}

impl Diurnal {
    fn tx_rx(&self, batch_size: usize, sender: &mut Sender<u64>, receiver: &mut Receiver<u64>) {
        // Send requests
        (0..batch_size).for_each(|_| while sender.send(1).is_err() {});

        // Wait for responses
        (0..batch_size).for_each(|_| while receiver.recv().is_err() {});
    }
}

impl LoadGen for Diurnal {
    fn run(&mut self, sender: &mut Sender<u64>, receiver: &mut Receiver<u64>) {
        let batch_sizes = [8, 16, 32, 16, 8];
        for batch_size in batch_sizes.iter() {
            let start = std::time::Instant::now();
            while start.elapsed().as_secs() < RECONFIG_TIMEOUT.as_secs() {
                self.tx_rx(*batch_size, sender, receiver);
            }
            self.tx_rx(*batch_size, sender, receiver);
        }
    }
}

pub struct Poisson {
    pub lambda: f64,
}

impl Poisson {
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }
}

impl Default for Poisson {
    fn default() -> Self {
        Self { lambda: 16.0 }
    }
}

impl Display for Poisson {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Poisson")
    }
}

impl LoadGen for Poisson {
    fn run(&mut self, sender: &mut Sender<u64>, receiver: &mut Receiver<u64>) {
        let start = std::time::Instant::now();
        let dist = rand::distributions::Poisson::new(self.lambda);
        let rng = &mut rand::thread_rng();

        // Send requests for 100 seconds
        while start.elapsed().as_secs() < 5 * RECONFIG_TIMEOUT.as_secs() {
            let batch_size = dist.sample(rng);
            (0..batch_size).for_each(|_| while sender.send(1).is_err() {});
            (0..batch_size).for_each(|_| while receiver.recv().is_err() {});
        }
    }
}
