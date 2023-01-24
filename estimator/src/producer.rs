use std::sync::mpsc::Sender;

use crate::loadgen::LoadGen;

pub struct Producer {
    sender: Sender<u64>,
}

impl Producer {
    pub fn new(sender: Sender<u64>) -> Self {
        Self { sender }
    }

    pub fn produce(&mut self, loadgen: &mut dyn LoadGen) {
        loadgen.run(&mut self.sender);
    }
}
