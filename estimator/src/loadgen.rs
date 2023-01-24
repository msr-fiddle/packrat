use std::sync::mpsc::Sender;

pub trait LoadGen {
    fn run(&mut self, _sender: &mut Sender<u64>) {}
}

pub struct Diurnal {
    pub start_time: u64,
    pub end_time: u64,
    pub num_requests: u64,
    pub request_rate: u64,
}

impl Diurnal {
    pub fn new(start_time: u64, end_time: u64, num_requests: u64, request_rate: u64) -> Self {
        Self {
            start_time,
            end_time,
            num_requests,
            request_rate,
        }
    }
}

impl LoadGen for Diurnal {
    fn run(&mut self, sender: &mut Sender<u64>) {
        let mut i = 0;
        loop {
            if i < self.num_requests {
                sender.send(i).unwrap();
                i += 1;
            } else {
                break;
            }
        }
    }
}
