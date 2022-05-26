#
# Copyright 2022 Intel Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from pyhdk._common cimport CSystemParameters
from pyhdk._storage cimport CSchemaProviderPtr, CDataProvider, CDataMgr, CBufferProvider
from pyhdk._execute cimport CExecutor, CResultSetPtr, CCompilationOptions, CExecutionOptions, CTargetMetaInfo

cdef extern from "omniscidb/QueryEngine/ExtensionFunctionsWhitelist.h":
  cdef cppclass CExtensionFunction "ExtensionFunction":
    pass

  cdef cppclass CExtensionFunctionsWhitelist "ExtensionFunctionsWhitelist":
    @staticmethod
    void add(const string&)

    @staticmethod
    void addUdfs(const string)

cdef extern from "omniscidb/QueryEngine/TableFunctions/TableFunctionsFactory.h" namespace "table_functions":
  cdef cppclass CTableFunction "table_functions::TableFunction":
    pass

  cdef cppclass CTableFunctionsFactory "table_functions::TableFunctionsFactory":
    @staticmethod
    void init()

    @staticmethod
    vector[CTableFunction] get_table_funcs(bool)

cdef extern from "omniscidb/Calcite/CalciteJNI.h":
  cdef cppclass FilterPushDownInfo:
    int input_prev;
    int input_start;
    int input_next;

  cdef cppclass CalciteJNI:
    CalciteJNI(CSchemaProviderPtr, string, size_t);
    string process(string, string, string, vector[FilterPushDownInfo], bool, bool, bool) except +

    string getExtensionFunctionWhitelist()
    string getUserDefinedFunctionWhitelist()
    void setRuntimeExtensionFunctions(const vector[CExtensionFunction]&, const vector[CTableFunction]&, bool)

cdef extern from "omniscidb/QueryEngine/RelAlgDagBuilder.h":
  cdef cppclass CRelAlgDag "RelAlgDag":
    pass

  cdef cppclass CRelAlgDagBuilder "RelAlgDagBuilder"(CRelAlgDag):
    CRelAlgDagBuilder(const string&, int, CSchemaProviderPtr) except +

cdef extern from "omniscidb/QueryEngine/Descriptors/RelAlgExecutionDescriptor.h":
  cdef cppclass CExecutionResult "ExecutionResult":
    CExecutionResult()
    CExecutionResult(const CExecutionResult&)
    CExecutionResult(CExecutionResult&&)

    const CResultSetPtr& getRows()
    const vector[CTargetMetaInfo]& getTargetsMeta()

cdef extern from "omniscidb/QueryEngine/RelAlgExecutor.h":
  cdef cppclass CRelAlgExecutor "RelAlgExecutor":
    CRelAlgExecutor(CExecutor*, CSchemaProviderPtr, CDataProvider*, unique_ptr[CRelAlgDag])

    CExecutionResult executeRelAlgQuery(const CCompilationOptions&, const CExecutionOptions&, const bool) except +
