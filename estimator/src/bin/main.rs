use std::sync::mpsc::channel;

use estimator::consumer::Consumer;
use estimator::loadgen::Diurnal;
use estimator::producer::Producer;

fn main() {
    env_logger::init();
    let (req_tx, req_rx) = channel::<u64>();
    let (resp_tx, resp_rx) = channel::<u64>();

    // Spawn a thread to produce a value
    let p = std::thread::spawn(move || {
        let mut producer = Producer::new(req_tx, resp_rx);
        producer.produce(&mut Diurnal::default());
    });

    // Spawn a thread to consume a value
    let c = std::thread::spawn(move || {
        let mut consumer = Consumer::new(req_rx, resp_tx);
        loop {
            consumer.consume();
        }
    });

    // Wait for the threads to finish
    p.join().unwrap();
    c.join().unwrap();
}
