
1. Follow README.md in the root directory to download and configure the code.

2. In the Quokka build directory, compile the test problem

```bash
# make sure you are in the build directory

ninja -j8 paper3_linear_diffusion
ninja src/paper3-linear-diffusion/test 
```

3. The output figures are located at `$QUOKKA_PATH/tests/LinearDiffusionMP_*.pdf`. You may copy them into this folder:

```bash
QUOKKA_PATH="/path/to/quokka"
WORKSPACE="/path/to/this/workspace"
cd $WORKSPACE
cd figure_scripts/fig3-linear-wave
# mkdir -p data/generated
# cp $QUOKKA_PATH/tests/LinearDiffusionMP_nx*.csv data/generated
cp $QUOKKA_PATH/tests/LinearDiffusionMP_*.pdf .
```
