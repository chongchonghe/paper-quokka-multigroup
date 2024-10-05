
## Make plots with precalculated data

```bash
python main.py
```

## Run QUOKKA tests and generate plots from the data

1. Follow README.md in the root directory to download and configure the code.

2. In the Quokka build directory, compile the test problem

```bash
# make sure you are in the build directory

ninja -j8 paper3_shock_1bin
ninja -j8 paper3_shock_32bin 
ninja src/paper3-shock/test 
```

3. Copy the output into `data/generated`

```bash
QUOKKA_PATH="/path/to/quokka"
WORKSPACE="/path/to/this/workspace"
cd $WORKSPACE
cd figure_scripts/fig5-shockMG
mkdir -p data/generated
cp $QUOKKA_PATH/tests/radshock_*.csv data/generated
```

4. Make the plots using the data created by QUOKKA

```bash
python main.py --quokka
```
