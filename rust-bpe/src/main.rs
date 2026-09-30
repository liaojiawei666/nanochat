mod bpe;
mod commands;
mod dataset;
mod paths;
mod special;

fn main() {
    let mut args = std::env::args().skip(1);
    let cmd = args
        .next()
        .unwrap_or_else(|| panic!("usage: rust-bpe <train|encode>"));
    if args.next().is_some() {
        panic!("usage: rust-bpe <train|encode>");
    }
    match cmd.as_str() {
        "train" => commands::train::run(),
        "encode" => commands::encode::run(),
        other => panic!("unknown command {other}, expected train or encode"),
    }
}
