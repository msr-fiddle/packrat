use std::sync::mpsc::Sender;

pub trait LoadGen {
    fn run(&mut self, _sender: &mut Sender<u64>) {}
}

pub struct Diurnal {
    pub num_requests: u64,
}

impl Diurnal {
    pub fn new(num_requests: u64) -> Self {
        Self { num_requests }
    }
}

impl LoadGen for Diurnal {
    fn run(&mut self, sender: &mut Sender<u64>) {
        let mut i = 0;
        while i < self.num_requests {
            sender.send(i).unwrap();
            i += 1;
        }
    }
}
