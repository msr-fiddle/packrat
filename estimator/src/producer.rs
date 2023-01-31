use std::sync::mpsc::{Receiver, Sender};

use crate::loadgen::LoadGen;

pub struct Producer {
    req_tx: Sender<u64>,
    resp_rx: Receiver<u64>,
}

impl Producer {
    pub fn new(req_tx: Sender<u64>, resp_rx: Receiver<u64>) -> Self {
        Self { req_tx, resp_rx }
    }

    pub fn produce(&mut self, loadgen: &mut dyn LoadGen) {
        loadgen.run(&mut self.req_tx, &mut self.resp_rx);
    }
}
