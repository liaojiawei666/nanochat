mod bpe;
mod commands;
mod dataset;
mod paths;
mod special;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("train") if args.len() == 1 => commands::train::run(),
        Some("encode") if args.len() == 1 => commands::encode::run(),
        Some("encode-text") if args.len() == 2 => {
            use std::io::Read;
            let encoder = bpe::BpeEncoder::from_tiktoken_path(std::path::Path::new(&args[1]));
            let mut text = String::new();
            std::io::stdin().read_to_string(&mut text).expect("read input");
            println!("{}", serde_json::to_string(&encoder.encode(&text)).unwrap());
        }
        _ => panic!("usage: rust-bpe <train|encode> OR rust-bpe encode-text <vocab.tiktoken>"),
    }
}
