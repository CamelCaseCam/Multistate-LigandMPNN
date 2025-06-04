# Docker Testing Env For Multistate-LigandMPNN Training

# Running tests

This command builds a docker image with the code of this repository and runs the repository's tests

```sh
./build_docker.sh mmpnn
docker run -t mmpnn ./run_tests.sh
```

```
[+] Building 1.1s (12/12) FINISHED                                                                                                                                           docker:default
 => [internal] load build definition from Dockerfile                                                                                                                                   0.0s
 => => transferring dockerfile: 438B                                                                                                                                                   0.0s
 => [internal] load metadata for docker.io/library/python:3.11-slim                                                                                                                    0.3s
 => [internal] load .dockerignore                                                                                                                                                      0.0s
 => => transferring context: 2B                                                                                                                                                        0.0s
 => [1/7] FROM docker.io/library/python:3.11-slim@sha256:9c85d1d49df54abca1c5db3b4016400e198e9e9bb699f32f1ef8e5c0c2149ccf                                                              0.0s
 => [internal] load build context                                                                                                                                                      0.0s
 => => transferring context: 54.52kB                                                                                                                                                   0.0s
 => CACHED [2/7] RUN apt-get update && apt-get install -y --no-install-recommends       build-essential       python3-dev       libblas-dev       liblapack-dev       && rm -rf /var/  0.0s
 => CACHED [3/7] WORKDIR /app                                                                                                                                                          0.0s
 => CACHED [4/7] RUN pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu                                                                                         0.0s
 => CACHED [5/7] COPY requirements.txt .                                                                                                                                               0.0s
 => CACHED [6/7] RUN pip install --no-cache-dir -r requirements.txt                                                                                                                    0.0s
 => [7/7] COPY . ./                                                                                                                                                                    0.3s
 => exporting to image                                                                                                                                                                 0.5s
 => => exporting layers                                                                                                                                                                0.5s
 => => writing image sha256:4263da4ca9498602aeaf6edfee8fb21da80a5ddc72b54eeb63a302939eb63cd9                                                                                           0.0s
 => => naming to docker.io/library/multimodel_mpnn_cr                                                                                                                                  0.0s

@> ProDy is configured: verbosity='none'
.....
----------------------------------------------------------------------
Ran 5 tests in 2.195s

OK
```

Note that this could take a while, as it needs to build ProDy from source. This command specifically installs the CPU version of PyTorch for compatibility, so please be patient if the tests run slowly. They shouldn't take more than a minute to complete on *any* hardware.

Further, note that while this code depends on the `requests` library to download PDB files for training, this library is not used for testing.

# Running a specific test

This example runs a single test in the class TestDiffusion, with the name "test_train_step"

```sh
./build_docker.sh mmpnn
docker run -t mmpnn ./run_tests.sh TestDiffusion.test_train_step
```
