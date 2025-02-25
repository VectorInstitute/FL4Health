# Documentation

This section of the repository contains source artifacts for building the docs
for `fl4health`.

What follow's next in this README is a guide for those who are interested in
building and serving the docs locally. You may be interested to do so if you are
contributing to `fl4health` and need to make appropriate changes to the
documentation.

## Build Docs

In order to build the docs, go into the `docs` directory and run the command
below:

```sh
cd docs/
make serve
```

The above command will build the docs as well as serve them locally, watching
for changes and presenting them in real-time, which is great for development!

Building the docs will take a couple of minutes, but once completed they will be
served on `http://127.0.0.1:8000`. Enter this address in your browser of choice
to see the docs.

> [!NOTE]
> We use sphinx to build our docs and api reference. Adjustments to style and
> and additional extensions are done via the appropriate modifications in
> `docs/source/conf.py`.
