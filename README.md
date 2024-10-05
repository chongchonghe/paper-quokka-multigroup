# Project Title


1. Follow the instruction on the [official QUOKKA documentation](https://quokka-astro.github.io/quokka/installation/) to install QUOKKA. You also want to checkout to the `chong/paper3-multigroup` branch before compilation. On my macOS, for example, the precedure is:

```bash
QUOKKA="/path/to/quokka"
git clone --recursive https://github.com/quokka-astro/quokka.git $QUOKKA
cd $QUOKKA
# checkout to branch chong/paper3-multigroup
git checkout -t chong/paper3-multigroup
# update submodules to match with the branch
git submodule update
# build the code
builddir="build/build-paper3-multigroup"
mkdir -p $builddir
cd $builddir
cmake ../.. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=-ffp-exception-behavior=maytrap -DAMReX_SPACEDIM=1 -G Ninja
# compile a test problem (with 8 cores)
ninja -j8 paper3_shock_1bin
```

Loc A: 0.1
Loc B: (0.01305 + 0.0005) / 0.01575 = 0.8603174603
