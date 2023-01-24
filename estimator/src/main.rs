use std::sync::mpsc::channel;

mod consumer;
mod producer;

use consumer::Consumer;
use producer::Producer;

fn main() {
    let (producer, consumer) = channel::<u64>();

    // Spawn a thread to produce a value
    let p = std::thread::spawn(move || {
        let mut producer = Producer::new(producer);
        producer.produce();
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
