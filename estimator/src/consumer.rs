use log::*;
use std::{
    cmp::min,
    collections::VecDeque,
    sync::mpsc::{Receiver, Sender},
    time::{Duration, Instant},
};

#[cfg(not(test))]
use std::{fs::OpenOptions, io::Write, path::Path};

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
    workload: String,
}

impl Consumer {
    pub fn new(workload: String, receiver: Receiver<u64>, sender: Sender<u64>) -> Self {
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
            workload,
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
            self.update_bs();
            self.last_reconfig_time = Instant::now();
            self.last_n_queue_size.clear();
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
                let mut old_bs = self.current_bs;
                let mut new_bs = self.current_bs;
                let mut transient_bs = new_bs as f64;

                for len in self.last_n_queue_size.iter() {
                    let residual_queue_len = (*len - self.current_bs as i64) as f64;
                    let bs_adjust = transient_bs * 0.3 + residual_queue_len * 0.7;
                    transient_bs = self.current_bs as f64 + bs_adjust;
                    debug!("Expected new batch size: {}", transient_bs);

                    new_bs = lower_power_of_two(transient_bs as i64) as usize;

                    info!(
                        "{:?},{},{},{},{:.2},{}",
                        self.counter,
                        *len,
                        self.current_bs,
                        residual_queue_len,
                        transient_bs,
                        new_bs
                    );

                    #[cfg(not(test))]
                    self.save_output(
                        self.counter,
                        *len,
                        self.current_bs,
                        residual_queue_len,
                        transient_bs as i64,
                        new_bs,
                    );
                    self.counter += 1;

                    // Reset transient state if batch size changes
                    if old_bs != new_bs {
                        old_bs = new_bs;
                        transient_bs = new_bs as f64;
                    }
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

    #[cfg(not(test))]
    fn save_output(
        &self,
        counter: usize,
        requests: i64,
        current_bs: usize,
        residual_queue_len: f64,
        bs: i64,
        new_bs: usize,
    ) {
        let benchmark = self.workload.to_lowercase();
        let file_name = format!("{benchmark}_benchmark.csv");

        let write_headers = !Path::new(&file_name).exists();
        let mut csv_file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(file_name)
            .expect("Can't open file");
        if write_headers {
            let row = "workload,counter,requests,current_bs,residual_queue_len,bs,new_bs\n";
            csv_file.write_all(row.as_bytes()).unwrap();
        }

        csv_file
            .write_all(
                format!(
                    "{},{},{},{},{},{},{}\n",
                    benchmark, counter, requests, current_bs, residual_queue_len, bs, new_bs
                )
                .as_bytes(),
            )
            .unwrap();
        csv_file.flush().expect("Can't flush file");
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
        let mut consumer = Consumer::new(String::from("test"), req_channel.1, res_channel.0);
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
        let mut consumer = Consumer::new(String::from("test"), req_channel.1, res_channel.0);
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
        let mut consumer = Consumer::new(String::from("test"), req_channel.1, res_channel.0);
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
        let mut consumer = Consumer::new(String::from("test"), req_channel.1, res_channel.0);
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
        let mut consumer = Consumer::new(String::from("test"), req_channel.1, res_channel.0);
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
