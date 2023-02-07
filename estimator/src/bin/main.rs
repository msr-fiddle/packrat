use clap::Parser;
use estimator::consumer::Consumer;
use estimator::loadgen::{Constant, Diurnal, LoadGen, Poisson};
use estimator::producer::Producer;
use std::sync::mpsc::channel;

#[derive(Parser)]
struct Args {
    #[arg(value_enum, short = 'w', long = "workload", default_value = "diurnal")]
    workload: Workload,
}

#[derive(clap::ValueEnum, PartialEq, Debug, Clone)]
pub enum Workload {
    Constant,
    Diurnal,
    Poisson,
}

fn get_workload(workload: Workload) -> Box<dyn LoadGen> {
    match workload {
        Workload::Constant => Box::<Constant>::default() as Box<dyn LoadGen>,
        Workload::Diurnal => Box::<Diurnal>::default() as Box<dyn LoadGen>,
        Workload::Poisson => Box::<Poisson>::default() as Box<dyn LoadGen>,
    }
}

fn main() {
    let workload = Args::parse().workload;
    println!("Running Batch Size Estimator for Workload: {:?}", workload);

    env_logger::init();
    let (req_tx, req_rx) = channel::<u64>();
    let (resp_tx, resp_rx) = channel::<u64>();

    // Spawn a thread to produce a value
    let p = std::thread::spawn(move || {
        let mut producer = Producer::new(req_tx, resp_rx);
        producer.produce(get_workload(workload).as_mut());
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
