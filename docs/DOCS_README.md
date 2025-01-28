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
make html
```

> [!NOTE]
> We use sphinx to build our docs and api reference. Adjustments to style and
> and additional extensions are done via the appropriate modifications in
> `docs/source/conf.py`.
