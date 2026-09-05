# OnlineTE

> **Note To The SIGCOMM Artifact Evaluation Committee**
>
> Testing `OnlineTE` requries multiple machines and if possible, a Gruobi license to validate the results.
> We deployed our solution on the `SPHERE` testbed, which we have used for larger (e.g. KDL level) tests.
> If testbed access is unavailable, then please reach out and we will try to open access to our available
> model, otherwise, we provide instructions for local testing.

----

## For Local Testing
Local testing will be slow. `OnlineTE` has a GPU backend, but it was not
evaluated as part of the paper, hence this branch only provides generic CPU-based implementation with parallelism over many nodes.

Instructions on how to test `OnlineTE` can be found under `LOCAL_TEST.md`.

## For Testing on `SPHERE`

We utilized [SPHERE](https://launch.sphere-testbed.net/) for our large scale tests. The testbed can quickly materialize networks of hundreds of VMs (although, at the time of writing, it has trouble implementing traffic shaping, hence delays are not reflected correctly).

We will release our experiment model as a public SPHERE artifact. Instructions on how to evaluate on that model can be found in `SPHERE_TEST.md`.

> If there are points of confusion, you may find some answers under `FAQ.md`, if not, please do not hesitate to reach out by opening an issue or emailing us!