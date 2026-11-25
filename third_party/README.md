# nanochat reference

The optimizer math and CORE prompt/scoring utilities are adapted from
[karpathy/nanochat](https://github.com/karpathy/nanochat) at commit
`92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` under the accompanying MIT license.
This project retains its own model architecture and uses FP32 optimizer storage.

Tests use an independent small mathematical reference and a fixture derived from
the same licensed optimizer equations. No upstream distributed implementation is
required at runtime.
