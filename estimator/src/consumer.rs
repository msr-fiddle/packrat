use log::*;
use std::{
    cmp::min,
    collections::VecDeque,
    sync::mpsc::Receiver,
    time::{Duration, Instant},
};

use crate::utils::{delay_loop, lower_power_of_two};

pub struct Consumer {
    receiver: Receiver<u64>,
    request_queue: VecDeque<u64>,
    last_reconfig_time: Instant,
    reconfig_timeouts: Duration,
    current_bs: usize,
    last_n_queue_size: Vec<i64>,
}

impl Consumer {
    pub fn new(receiver: Receiver<u64>) -> Self {
        let request_queue = VecDeque::with_capacity(1024);
        Self {
            receiver,
            request_queue,
            last_reconfig_time: Instant::now(),
            reconfig_timeouts: Duration::from_secs(100),
            current_bs: 1,
            last_n_queue_size: Vec::with_capacity(10),
        }
    }

    pub fn consume(&mut self) {
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(5) {
            if let Ok(value) = self.receiver.recv_timeout(Duration::from_millis(1_u64)) {
                self.request_queue.push_back(value);
            }
        }
        println!("Received {} requests", self.request_queue.len());
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
        let single_request_delay = 1;
        let process = min(self.request_queue.len(), self.current_bs);

        // Process `process` number of requests in the queue
        // TODO: delay using profiling data
        for _ in 0..process {
            let _drop = self.request_queue.pop_front();
            delay_loop(single_request_delay);
        }
        info!("Processed {} requests", process);
    }

    fn update_bs(&mut self) {
        match self.last_n_queue_size.iter().sum::<i64>() {
            0 => self.current_bs = 0,
            _ => {
                let last_n_queue_sum = self.last_n_queue_size.iter().sum::<i64>();
                let average_queue_size = last_n_queue_sum / self.last_n_queue_size.len() as i64;
                let expected_bs =
                    (self.current_bs as f64 * 0.00 + average_queue_size as f64 * 1.0) as i64;
                println!("Expected batch size: {}", expected_bs);

                let new_bs = match expected_bs {
                    x if x < 0 => lower_power_of_two(self.current_bs as i64 + x) as usize,
                    x if x > 0 => lower_power_of_two(x) as usize,
                    x => x as usize,
                };

                if new_bs != self.current_bs {
                    self.current_bs = new_bs;
                    println!("Current batch size: {}", self.current_bs);
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
        let channel = channel();
        let mut consumer = Consumer::new(channel.1);
        assert_eq!(consumer.current_bs, 1);

        (0..100).for_each(|_| consumer.last_n_queue_size.push(16));
        assert_eq!(consumer.last_n_queue_size.len() as i64, 100);
        assert_eq!(consumer.last_n_queue_size.iter().sum::<i64>(), 1600);

        consumer.update_bs();
        assert_eq!(consumer.current_bs, 16);
    }

    #[test]
    fn test_bs_decrease() {
        let channel = channel();
        let mut consumer = Consumer::new(channel.1);
        assert_eq!(consumer.current_bs, 1);
        consumer.current_bs = 32;

        (0..100).for_each(|_| consumer.last_n_queue_size.push(-16));
        assert_eq!(consumer.last_n_queue_size.len() as i64, 100);
        assert_eq!(consumer.last_n_queue_size.iter().sum::<i64>(), -1600);

        consumer.update_bs();
        assert_eq!(consumer.current_bs, 16);
    }

    #[test]
    fn test_bs_no_change() {
        let channel = channel();
        let mut consumer = Consumer::new(channel.1);
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
        let channel = channel();
        let mut consumer = Consumer::new(channel.1);
        assert_eq!(consumer.current_bs, 1);
        consumer.reconfig_timeouts = Duration::from_secs(2);
        let new_bs = 16;

        let start = Instant::now();
        while consumer.current_bs == 1 {
            (0..new_bs).for_each(|_| channel.0.send(1).unwrap());
            consumer.consume();
        }
        assert!(start.elapsed() > consumer.reconfig_timeouts);
        assert_eq!(consumer.current_bs, 16);
    }

    #[test]
    fn test_bs_approach_to_zero() {
        let channel = channel();
        let mut consumer = Consumer::new(channel.1);
        assert_eq!(consumer.current_bs, 1);
        consumer.reconfig_timeouts = Duration::from_secs(2);
        let new_bs = 16;

        let start = Instant::now();
        while consumer.current_bs == 1 {
            (0..new_bs).for_each(|_| channel.0.send(1).unwrap());
            consumer.consume();
        }
        assert!(start.elapsed() > consumer.reconfig_timeouts);
        assert_eq!(consumer.current_bs, 16);

        let start = Instant::now();
        let new_bs = 1;
        while consumer.current_bs != 1 {
            (0..new_bs).for_each(|_| channel.0.send(1).unwrap());
            consumer.consume();
            if start.elapsed() > consumer.reconfig_timeouts * 2 {
                break;
            }
        }
        assert!(start.elapsed() > consumer.reconfig_timeouts);
        assert_eq!(consumer.current_bs, 1);
    }
}
