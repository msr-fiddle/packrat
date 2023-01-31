use std::sync::mpsc::{Receiver, Sender};

use crate::RECONFIG_TIMEOUT;

pub trait LoadGen {
    fn run(&mut self, _sender: &mut Sender<u64>, _receiver: &mut Receiver<u64>) {}
}

#[derive(Debug, Default)]
pub struct Diurnal;

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
        let starting_bs = 16;
        let mut start = std::time::Instant::now();

        // Send requests for 100 seconds
        while start.elapsed().as_secs() < RECONFIG_TIMEOUT.as_secs() {
            self.tx_rx(starting_bs, sender, receiver);
        }

        start = std::time::Instant::now();
        while start.elapsed().as_secs() < RECONFIG_TIMEOUT.as_secs() {
            self.tx_rx(starting_bs * 2, sender, receiver);
        }
    }
}
