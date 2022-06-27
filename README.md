# oneAPI Heterogeneous Data Kernels
oneHDK is a low-level execution library for data analytics processing. 

HDK is used as a fast execution backend in [Modin](https://github.com/intel-ai/modin). The HDK library provides a set of components for federating analytic queries to an execution backend based on [OmniSciDB](https://github.com/intel-ai/omniscidb). Currently, HDK targets OLAP-style queries expressed as relational algebra or SQL.  The APIs required for Modin support have been exposed in a library installed from this repository, `pyhdk`. Major and immediate project priorities include:
- Introducing a HDK-specific IR and set of optimizations to reduce reliance on RelAlg and improve extensibility of the query API. 
- Supporting heterogeneous device execution, where a query is split across a set of hardware devices (e.g. CPU and GPU) for best performance. We have developed an initial cost model for heterogeneous execution.
- Improving performance of the CPU backend on Modin-specific queries and current-generation data science workstations and servers by > 2x. 

We are committed to supporting a baseline set of functionality on all x86 CPUs, later-generation NVIDIA GPUs (supporting CUDA 11+), and Intel GPUs. The x86 backend uses LLVM ORCJIT for x86 byte code generation. The NVIDIA backend uses NVPTX extensions in LLVM to generate PTX, which is JIT-compiled by the CUDA runtime compiler. The Intel GPU backend leverages the LLVM [SPIR-V translator](https://github.com/KhronosGroup/SPIRV-LLVM-Translator) to produce SPIR-V. Device code is generated using the [Intel Graphics Compiler (IGC)](https://github.com/intel/intel-graphics-compiler) via the [oneAPI L0 driver](https://github.com/oneapi-src/level-zero).

## Components

### Storage

`ArrowStorage` is currently the default (and only available) HDK storage layer. `ArrowStorage` provides storage support for [Apache Arrow](https://github.com/apache/arrow) format data. The storage layer must be explicitly initialized:

```
import pyhdk
storage = pyhdk.storage.ArrowStorage(1)
```

The parameter applied to the `ArrowStorage` constructor is the database ID. The database ID allows storage instances to be kept logically separate.

`ArrowStorage` automatically converts Arrow format datatypes to `omniscidb` datatypes. Some variable length types are not yet supported, but scalar types are available. `pyarrow` can be used to convert Pandas DataFrames to Arrow:

```
       at = pyarrow.Table.from_pandas(
            pandas.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
        )
```

The arrow table can then be imported using the Arrow storage interface.

```
        opt = pyhdk.storage.TableOptions(2)
        storage.importArrowTable(at, "test", opt)
```

### Data Manager

The Data Manager controls the storage and in-memory buffer pools for all queries. Storage engines must be registered with the data manager:

```
        data_mgr = pyhdk.storage.DataMgr()
        data_mgr.registerDataProvider(storage)
```

### Query Execution

Three high level components are required to execute a query:

1. Calcite: This is a wrapper around [Apache Calcite](https://calcite.apache.org/) handling SQL parsing and relational algebra optimization. Queries are first sent to Calcite for parsing and conversion to relational algebra. Depending on the query, some optimization of the relational algebra occurs in Calcite.
2. RelAlgExecutor: Handles execution of a relational algebra tree. Only one should be created per query. 
3. Executor: The JIT compilation and query execution engine. Holds state which spans queries (e.g. code cache). Should be created as a singleton and re-used per query. 

The complete flow is as follows:

```
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
```
     df = res.to_arrow().to_pandas()
```

## Build

HDK includes components from external libraries as submodules. To clone the project, either use `git clone --recurse-submodules` to initially clone the repo, or clone without using flags and then run:

```
git submodule init
git submodule update
```

When pulling changes from upstream, be sure to update submodule references using `git submodule update`.

If using a Conda enviornment, run the following to build and install HDK:

```
mkdir build
cd build
cmake -DENABLE_CONDA=on ..
make -j
make install
```

### Dependencies 

Conda environments are used for HDK development. 

## Test

Python tests can be run from the python source directory using `pytest`. 
