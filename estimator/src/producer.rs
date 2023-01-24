use std::sync::mpsc::Sender;

pub struct Producer {
    sender: Sender<u64>,
}

impl Producer {
    pub fn new(sender: Sender<u64>) -> Self {
        Self { sender }
    }

    pub fn produce(&mut self) {
        self.sender.send(1).unwrap();
    }
}
