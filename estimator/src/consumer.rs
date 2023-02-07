use log::*;
use std::{
    cmp::min,
    collections::VecDeque,
    sync::mpsc::{Receiver, Sender},
    time::{Duration, Instant},
};

use crate::{
    utils::{delay_loop, lower_power_of_two},
    RECONFIG_TIMEOUT, SINGLE_REQUEST_DELAY,
};

pub struct Consumer {
    receiver: Receiver<u64>,
    sender: Sender<u64>,
    request_queue: VecDeque<u64>,
    last_reconfig_time: Instant,
    reconfig_timeouts: Duration,
    current_bs: usize,
    last_n_queue_size: Vec<i64>,
    counter: usize,
}

impl Consumer {
    pub fn new(receiver: Receiver<u64>, sender: Sender<u64>) -> Self {
        let request_queue = VecDeque::with_capacity(1024);
        Self {
            receiver,
            sender,
            request_queue,
            last_reconfig_time: Instant::now(),
            reconfig_timeouts: RECONFIG_TIMEOUT,
            current_bs: 1,
            last_n_queue_size: Vec::with_capacity(10),
            counter: 0,
        }
    }

    pub fn consume(&mut self) {
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(5) {
            if let Ok(value) = self.receiver.recv_timeout(Duration::from_millis(1_u64)) {
                self.request_queue.push_back(value);
            }
        }
        debug!("Received {} requests", self.request_queue.len());
        self.last_n_queue_size.push(self.request_queue.len() as i64);

        match self.request_queue.len() {
            x if x < self.current_bs => self.handle_request(),
            _ => {
                while self.request_queue.len() >= self.current_bs {
                    self.handle_request();
                }
            }
        }

        // Is it time to reconfigure?
        if self.last_reconfig_time.elapsed() > self.reconfig_timeouts {
            let old_bs = self.current_bs;
            self.update_bs();

            if old_bs != self.current_bs {
                info!("Reconfigured BS from {} to {}", old_bs, self.current_bs);
                self.last_reconfig_time = Instant::now();
                self.last_n_queue_size.clear();
            }
        }
    }

    fn handle_request(&mut self) {
        let process = min(self.request_queue.len(), self.current_bs);

        // Process `process` number of requests in the queue
        // TODO: delay using profiling data
        for _ in 0..process {
            let _drop = self.request_queue.pop_front();
            delay_loop(SINGLE_REQUEST_DELAY);
            self.sender.send(1).unwrap();
        }
        debug!("Processed {} requests", process);
    }

    fn update_bs(&mut self) {
        match self.last_n_queue_size.iter().sum::<i64>() {
            0 => self.current_bs = 0,
            _ => {
                let mut new_bs = self.current_bs;
                let mut transient_bs = new_bs as f64;

                for len in self.last_n_queue_size.iter() {
                    let residual_queue_len = (*len - self.current_bs as i64) as f64;
                    transient_bs = transient_bs * 0.1 + residual_queue_len * 0.9;
                    debug!("Expected new batch size: {}", transient_bs);

                    let bs = self.current_bs as i64 + transient_bs as i64;
                    new_bs = lower_power_of_two(bs) as usize;

                    info!(
                        "{:?},{},{},{},{}",
                        self.counter, self.current_bs, residual_queue_len, bs, new_bs
                    );
                    self.counter += 1;
                }

                if new_bs != self.current_bs {
                    println!(">>> Updated BS from {} to {}", self.current_bs, new_bs);
                    self.current_bs = new_bs;
                    assert!(self.current_bs.is_power_of_two());
                }
            }
        }

        if self.current_bs == 0 {
            println!("Batch size is 0. Exiting...");
            std::process::exit(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    #[test]
    fn test_bs_increase() {
        let req_channel = channel();
        let res_channel = channel();
        let mut consumer = Consumer::new(req_channel.1, res_channel.0);
        assert_eq!(consumer.current_bs, 1);

        (0..100).for_each(|_| consumer.last_n_queue_size.push(16));
        assert_eq!(consumer.last_n_queue_size.len() as i64, 100);
        assert_eq!(consumer.last_n_queue_size.iter().sum::<i64>(), 1600);

        consumer.update_bs();
        assert_eq!(consumer.current_bs, 16);
    }

    #[test]
    fn test_bs_decrease() {
        let req_channel = channel();
        let res_channel = channel();
        let mut consumer = Consumer::new(req_channel.1, res_channel.0);
        assert_eq!(consumer.current_bs, 1);
        consumer.current_bs = 32;

        (0..100).for_each(|_| consumer.last_n_queue_size.push(16));
        assert_eq!(consumer.last_n_queue_size.len() as i64, 100);
        assert_eq!(consumer.last_n_queue_size.iter().sum::<i64>(), 1600);

        consumer.update_bs();
        assert_eq!(consumer.current_bs, 16);
    }

    #[test]
    fn test_bs_no_change() {
        let req_channel = channel();
        let res_channel = channel();
        let mut consumer = Consumer::new(req_channel.1, res_channel.0);
        assert_eq!(consumer.current_bs, 1);
        consumer.current_bs = 32;

        (0..100).for_each(|_| consumer.last_n_queue_size.push(32));
        assert_eq!(consumer.last_n_queue_size.len() as i64, 100);
        assert_eq!(consumer.last_n_queue_size.iter().sum::<i64>(), 3200);

        consumer.update_bs();
        assert_eq!(consumer.current_bs, 32);
    }

    #[test]
    fn test_bs_increase_e2e() {
        let req_channel = channel();
        let res_channel = channel();
        let mut consumer = Consumer::new(req_channel.1, res_channel.0);
        assert_eq!(consumer.current_bs, 1);
        consumer.reconfig_timeouts = Duration::from_secs(2);
        let new_bs = 16;

        let start = Instant::now();
        while consumer.current_bs == 1 {
            (0..new_bs).for_each(|_| req_channel.0.send(1).unwrap());
            consumer.consume();
        }
        assert!(start.elapsed() > consumer.reconfig_timeouts);
        assert_eq!(consumer.current_bs, 16);
    }

    #[test]
    fn test_bs_approach_to_zero() {
        let req_channel = channel();
        let res_channel = channel();
        let mut consumer = Consumer::new(req_channel.1, res_channel.0);
        assert_eq!(consumer.current_bs, 1);
        consumer.reconfig_timeouts = Duration::from_secs(2);
        let new_bs = 16;

        let start = Instant::now();
        while consumer.current_bs == 1 {
            (0..new_bs).for_each(|_| req_channel.0.send(1).unwrap());
            consumer.consume();
        }
        assert!(start.elapsed() > consumer.reconfig_timeouts);
        assert_eq!(consumer.current_bs, 16);

        let start = Instant::now();
        let new_bs = 1;
        while start.elapsed() < consumer.reconfig_timeouts * 2 {
            (0..new_bs).for_each(|_| req_channel.0.send(1).unwrap());
            consumer.consume();
        }
        assert!(start.elapsed() > consumer.reconfig_timeouts);
        assert_eq!(consumer.current_bs, 1);
    }
}
