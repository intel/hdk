# HDK - Heterogeneous Data Kernels      [![Conda Version](https://img.shields.io/conda/vn/conda-forge/pyhdk.svg)](https://anaconda.org/conda-forge/pyhdk)
HDK is a low-level execution library for data analytics processing. 

HDK is used as a fast execution backend in [Modin](https://github.com/intel-ai/modin). The HDK library provides a set of components for federating analytic queries to an execution backend based on [OmniSciDB](https://github.com/intel-ai/omniscidb). Currently, HDK targets OLAP-style queries expressed as relational algebra or SQL.  The APIs required for Modin support have been exposed in a library installed from this repository, `pyhdk`. Major and immediate project priorities include:
- Introducing a HDK-specific IR and set of optimizations to reduce reliance on RelAlg and improve extensibility of the query API. 
- Supporting heterogeneous device execution, where a query is split across a set of hardware devices (e.g. CPU and GPU) for best performance. We have developed an initial cost model for heterogeneous execution.
- Improving performance of the CPU backend on Modin-specific queries and current-generation data science workstations and servers by > 2x. 

We are committed to supporting a baseline set of functionality on all x86 CPUs, later-generation NVIDIA GPUs (supporting CUDA 11+), and Intel GPUs. The x86 backend uses LLVM ORCJIT for x86 byte code generation. The NVIDIA backend uses NVPTX extensions in LLVM to generate PTX, which is JIT-compiled by the CUDA runtime compiler. The Intel GPU backend leverages the LLVM [SPIR-V translator](https://github.com/KhronosGroup/SPIRV-LLVM-Translator) to produce SPIR-V. Device code is generated using the [Intel Graphics Compiler (IGC)](https://github.com/intel/intel-graphics-compiler) via the [oneAPI L0 driver](https://github.com/oneapi-src/level-zero).

## Components

### Config

`Config` controls library-wide properties and must be passed to `Executor` and `DataMgr`. Default config objects should suffice for most installations. Instantiate a config first as part of library setup.

### Storage

`ArrowStorage` is currently the default (and only available) HDK storage layer. `ArrowStorage` provides storage support for [Apache Arrow](https://github.com/apache/arrow) format data. The storage layer must be explicitly initialized:

```python
import pyhdk
storage = pyhdk.storage.ArrowStorage(1)
```

The parameter applied to the `ArrowStorage` constructor is the database ID. The database ID allows storage instances to *be* kept logically separate.

`ArrowStorage` automatically converts Arrow format datatypes to `omniscidb` datatypes. Some variable length types are not yet supported, but scalar types are available. `pyarrow` can be used to convert Pandas DataFrames to Arrow:

```python
at = pyarrow.Table.from_pandas(
    pandas.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
)
```

The arrow table can then be imported using the Arrow storage interface.

```python
opt = pyhdk.storage.TableOptions(2)
storage.importArrowTable(at, "test", opt)
```

### Data Manager

The Data Manager controls the storage and in-memory buffer pools for all queries. Storage engines must be registered with the data manager:

```python
data_mgr = pyhdk.storage.DataMgr()
data_mgr.registerDataProvider(storage)
```

### Query Execution

Three high level components are required to execute a query:

1. Calcite: This is a wrapper around [Apache Calcite](https://calcite.apache.org/) handling SQL parsing and relational algebra optimization. Queries are first sent to Calcite for parsing and conversion to relational algebra. Depending on the query, some optimization of the relational algebra occurs in Calcite.
2. RelAlgExecutor: Handles execution of a relational algebra tree. Only one should be created per query. 
3. Executor: The JIT compilation and query execution engine. Holds state which spans queries (e.g. code cache). Should be created as a singleton and re-used per query. 

The complete flow is as follows:

```python
calcite = pyhdk.sql.Calcite(storage)
executor = pyhdk.Executor(data_mgr)
ra = calcite.process("SELECT * FROM t;")
rel_alg_executor = pyhdk.sql.RelAlgExecutor(
    executor, storage, data_mgr, ra
)
res = rel_alg_executor.execute()
```

Calcite reads the schema information from storage, and the Executor stores a reference to Data Manager for buffer/storage access during a query. 

The return from RelAlgExecutor is a ResultSet object which can be converted to Arrow and to pandas:
```python
df = res.to_arrow().to_pandas()
```

## Examples

Standalone examples are available in the `examples` directory. Most examples run via Jupyter notebooks. 


## Build

### Dependencies 

Miniconda installation is required. (Anaconda may produce build issues.) Use one of these [miniconda installers](https://docs.conda.io/en/latest/miniconda.html).

Conda environments are used for HDK development. Use the YAML file in `omniscidb/scripts/`:

```bash
conda env create -f omniscidb/scripts/mapd-deps-conda-dev-env.yml
conda activate omnisci-dev
```

### Compilation

If using a Conda environment, run the following to build and install HDK:

```bash
mkdir build && cd build
cmake ..
make -j 
make install
```

By default GPU support is disabled.

To verify check `python -c 'import pyhdk'` executed without an error.

#### Compilation with Intel GPU support

##### Dependencies

Install extra dependencies into the existing environment:

```bash
conda install -c conda-forge level-zero-devel pkg-config
```

##### Compilation

```bash
mkdir build && cd build
cmake -DENABLE_L0=on ..
make -j 
make install
```

#### Compilation with CUDA support

##### Dependencies

Install extra dependencies into an existing environment or a new one.

```bash
conda install -c conda-forge cudatoolkit-dev arrow-cpp-proc=3.0.0=cuda arrow-cpp=11.0=*cuda
```

##### Compilation

```bash
mkdir build && cd build
cmake -DENABLE_CUDA=on ..
make -j 
make install
```

### Issues

If you meet issues during the build refer to [`.github/workflows/build.yml`](.github/workflows/build.yml). This file describes the compilation steps used for the CI build.

If you are still facing issues please create a github issue. 

## Test

Python tests can be run from the python source directory using `pytest`. 

### HDK interface tests

```bash
pytest python/tests/*.py 
```

### Modin integration tests

```bash
pytest python/tests/modin
```

### All pytests
```bash
pytest python/tests/ 
```

## (Optional dependency) Modin installation

Installation into conda environment. 

Clone [Modin](https://github.com/modin-project/modin). 

```bash
cd modin && pip install -e .
```

## Pytest logging 

To enable logging: 
```python
pyhdk.initLogger(debug_logs=True)
```

In the `setup_class(..)` body.

Logs are by default located in the `hdk_log/` folder. 
