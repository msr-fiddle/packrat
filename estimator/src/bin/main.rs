use std::sync::mpsc::channel;

use estimator::consumer::Consumer;
use estimator::loadgen::Diurnal;
use estimator::producer::Producer;

fn main() {
    env_logger::init();
    let (producer, consumer) = channel::<u64>();

    // Spawn a thread to produce a value
    let p = std::thread::spawn(move || {
        let mut producer = Producer::new(producer);
        producer.produce(&mut Diurnal::new(1000));
    });

    // Spawn a thread to consume a value
    let c = std::thread::spawn(move || {
        let mut consumer = Consumer::new(consumer);
        consumer.consume();
    });

    // Wait for the threads to finish
    p.join().unwrap();
    c.join().unwrap();
}
