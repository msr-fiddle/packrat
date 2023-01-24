use std::{cmp::min, collections::VecDeque, hint::spin_loop, sync::mpsc::Receiver, time::Duration};

pub struct Consumer {
    receiver: Receiver<u64>,
    request_queue: VecDeque<u64>,
    timeout: u128,
    num_timeouts: u64,
    current_bs: usize,
    last_n_queue_size: Vec<u64>,
}

impl Consumer {
    pub fn new(receiver: Receiver<u64>) -> Self {
        let request_queue = VecDeque::with_capacity(1024);
        Self {
            receiver,
            request_queue,
            timeout: 1000,
            num_timeouts: 0,
            current_bs: 1,
            last_n_queue_size: Vec::with_capacity(10),
        }
    }

    pub fn consume(&mut self) {
        let start = std::time::Instant::now();
        loop {
            let remaining = self.timeout - start.elapsed().as_millis();
            match self
                .receiver
                .recv_timeout(Duration::from_millis(remaining as u64))
            {
                Ok(value) => self.request_queue.push_back(value),
                Err(_) => match self.request_queue.len() >= self.current_bs {
                    // If the server receives >= current_bs requests, it will
                    // process them all and then reset the timeout counter.
                    true => {
                        self.last_n_queue_size.push(self.request_queue.len() as u64);
                        self.num_timeouts = 0;
                        self.handle_request();
                        self.update_bs();
                    }

                    // If the server receives < current_bs requests, it will
                    // process them all and increment the timeout counter.
                    false => {
                        self.last_n_queue_size.push(self.request_queue.len() as u64);
                        self.num_timeouts += 1;
                        self.handle_request();
                        self.update_bs();
                    }
                },
            }
        }
    }

    fn handle_request(&mut self) {
        let single_request_delay = 10;
        let process = min(self.request_queue.len(), self.current_bs);
        // Process `process` number of requests in the queue
        for _ in 0..process {
            let _drop = self.request_queue.pop_front();
            delay_loop(single_request_delay);
        }
    }

    fn update_bs(&mut self) {
        // Decrease batch size if timed out multiple times.
        // Otherwise, increased the batch size using EWMA.
        if self.num_timeouts >= 3 {
            self.current_bs = self.current_bs / 2;
        } else {
            let last_n_queue_size = self.last_n_queue_size.iter().sum::<u64>();
            let last_n_queue_size = last_n_queue_size as f64 / self.last_n_queue_size.len() as f64;
            let expected_bs = (self.current_bs as f64 * 0.9 + last_n_queue_size * 0.1) as usize;

            if expected_bs >= 2 * self.current_bs {
                self.current_bs = expected_bs;
            }
        }

        if self.current_bs == 0 {
            println!("Batch size is 0. Exiting...");
            std::process::exit(0);
        }
        assert!(self.current_bs.is_power_of_two());
    }
}

pub fn delay_loop(delay: u128) {
    let start = std::time::Instant::now();
    loop {
        spin_loop();
        let remaining = delay - start.elapsed().as_millis();
        if remaining <= 0 {
            break;
        }
    }
}
