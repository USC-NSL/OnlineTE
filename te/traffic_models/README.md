# Traffic Matrix Generators
There are no large scale, fine-grained, public Traffic Matrix (TM) data from real wide area networks (as they can reveal sensitive information about network operators). As such, we test using synthetic matrices.

- `base.py` defines the ABC of a traffic matrix generator objective (`TMGenerator`). OnlineTE is supposed to be evaluated on a whole _sequence_ of matrices rather than just one matrix at a time. The generator objects specifically create such a sequence for evaluating OnlineTE (or any other solver).

- `generators.py` contains the individual TM generators and their parameter definitions.

| Name | Description |
|------|-------------|
| Uniform | Samples TM entries from a uniform distribution |
| Exponential | Samples TM entries from an exponential distribution. This makes traffic between distant nodes less likely |
| Bimodal | Creates a distirbution with two distinct peaks to simulate two different classes of traffic (often large "elephant" flows and tiny "mouse" flows) |
| File-backed | Sequentially read matrices from a given directory |
| Uniform Drift | Starting from an arbitrary matrix, generate new ones by adding noise sampled from a uniform distribution |
| Sampled | Starting from an arbitrary matrix, generate new ones by randomly sampling a number of entries and arbitrarily changing them |
| Noisy-Sampled | Similar to `Sampled`, but adds a controlled noise to all other entries as well |
| NCFlow | A generator inspired by the one used in the `NCFlow` paper |
| Gravity | Generate a traffic matrix using the gravity model (nodes that send more data must receive more data as well) |