# simple_logger [![](https://img.shields.io/github/tag/borntyping/rust-simple_logger.svg)](https://github.com/borntyping/rust-simple_logger/tags)

A logger that prints all messages with a readable output format.

The output format is based on the format used by [Supervisord](https://github.com/Supervisor/supervisor), with timestamps in [RFC 3339](https://datatracker.ietf.org/doc/html/rfc3339) format.

The format and timezone used for timestamps can be customised, simple colours based on log level can be enabled, thread metadata can be included, and output can be toggled between STDOUT and STDERR. 

* [Source on GitHub](https://github.com/borntyping/rust-simple_logger)
* [Packages on Crates.io](https://crates.io/crates/simple_logger)
* [Documentation on Docs.rs](https://docs.rs/simple_logger)

Notices
-------

### Project status

I wrote the initial version of this library in 2015, and haven't written Rust professionally since then.
I consider this as close as I'll ever get to a "finished" project and don't plan on adding any more features to it.

It is still maintained and I try and merge pull requests for fixes, improvements, and features; though I generally turn down pull requests for large or complex features that go outside the library's aim to be a simple logger.
If you need more, the `log` module has a list of [available logging implementations](https://docs.rs/log/latest/log/#available-logging-implementations), or you could consider forking `simple_logger` and building on top of it.

### Breaking changes

- **Version 2.0.0 changes the default from displaying timestamps in the local timezone to displaying timestamps in UTC.** See issue [#52](https://github.com/borntyping/rust-simple_logger/issues/52) for more information.

Usage
-----

```rust
use simple_logger::SimpleLogger;

fn main() {
    SimpleLogger::new().init().unwrap();

    log::warn!("This is an example message.");
}
```

This outputs:

```txt
2022-01-19T17:27:07.013874956Z WARN [logging_example] This is an example message.
```

You can run the above example with:

```sh
cargo run --example init
```

### Optional features

The `colors` and `timestamps` features are enabled by default. You can remove these
features and their respective dependencies by disabling all features in your
`Cargo.toml`.

```toml
[dependencies.simple_logger]
default-features = false
```

To include the `timestamps` feature, but not the `colors` feature:

```toml
[dependencies.simple_logger]
default-features = false
features = ["timestamps"]
```

To include the `colors` feature, but not the `timestamps` feature:

```toml
[dependencies.simple_logger]
default-features = false
features = ["colors"]
```

To include thread metadata use the `threads` and `nightly` features:

```toml
[dependencies.simple_logger]
features = ["threads", "nightly"]
```

To direct logging output to `stderr` use the `stderr` feature:

```toml
[dependencies.simple_logger]
features = ["stderr"]
```

Multiple features can be combined.

```toml
[dependencies.simple_logger]
features = ["colors", "threads", "timestamps", "nightly", "stderr"]
```

### Wrapping with another logger

You might want to wrap this logger to do your own processing before handing events to a SimpleLogger instance. Instead
of calling `init()` which calls `log::set_max_level` and `log::set_boxed_logger`, you can call those functions directly
giving you the chance to wrap or adjust the logger. See [wrap.rs](examples/wrap.rs) for a more detailed example.

### Console setup

The `SimpleLogger.init()` function attempts to configure colours support as best it can in various situations:

- On Windows, it will enable colour output. _See `set_up_windows_color_terminal()`._
- When using the `colors` *and* `stderr` features, it will instruct the `colored` library to display colors if STDERR
  is a terminal (instead of checking if STDOUT is a terminal). _See `use_stderr_for_colors()`._

Licence
-------

`simple_logger` is licenced under the [MIT Licence](http://opensource.org/licenses/MIT).

Authors
-------

Written by [Sam Clements](sam@borntyping.co.uk).
