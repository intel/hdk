#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "omniscidb/Shared/sqltypes.h":
  enum CSQLTypes "SQLTypes":
    kNULLT = 0,
    kBOOLEAN = 1,
    kCHAR = 2,
    kVARCHAR = 3,
    kNUMERIC = 4,
    kDECIMAL = 5,
    kINT = 6,
    kSMALLINT = 7,
    kFLOAT = 8,
    kDOUBLE = 9,
    kTIME = 10,
    kTIMESTAMP = 11,
    kBIGINT = 12,
    kTEXT = 13,
    kDATE = 14,
    kARRAY = 15,
    kINTERVAL_DAY_TIME = 16,
    kINTERVAL_YEAR_MONTH = 17,
    kTINYINT = 18,
    kEVAL_CONTEXT_TYPE = 19,
    kVOID = 20,
    kCURSOR = 21,
    kCOLUMN = 22,
    kCOLUMN_LIST = 23,
    kSQLTYPE_LAST = 24,

  enum CEncodingType "EncodingType":
    kENCODING_NONE = 0,
    kENCODING_FIXED = 1,
    kENCODING_RL = 2,
    kENCODING_DIFF = 3,
    kENCODING_DICT = 4,
    kENCODING_SPARSE = 5,
    kENCODING_DATE_IN_DAYS = 7,
    kENCODING_LAST = 8,

  cdef cppclass CSQLTypeInfo "SQLTypeInfo":
    CSQLTypeInfo(CSQLTypes t, int d, int s, bool n, CEncodingType c, int p, CSQLTypes st)
    CSQLTypeInfo(CSQLTypes t, int d, int s, bool n)
    CSQLTypeInfo(CSQLTypes t, CEncodingType c, int p, CSQLTypes st)
    CSQLTypeInfo(CSQLTypes t, int d, int s)
    CSQLTypeInfo(CSQLTypes t, bool n)
    CSQLTypeInfo(CSQLTypes t)
    CSQLTypeInfo(CSQLTypes t, bool n, CEncodingType c)
    CSQLTypeInfo()

    CSQLTypes get_type()
    CSQLTypes get_subtype()
    int get_dimension()
    int get_precision()
    int get_input_srid()
    int get_scale()
    int get_output_srid()
    bool get_notnull()
    CEncodingType get_compression()
    int get_comp_param()
    int get_size()
    int get_logical_size()

    string toString()

cdef class TypeInfo:
  cdef CSQLTypeInfo c_type_info

cdef extern from "omniscidb/Shared/SystemParameters.h":
  cdef cppclass CSystemParameters "SystemParameters":
    CSystemParameters()

cdef extern from "omniscidb/ThriftHandler/CommandLineOptions.h":
  cdef bool g_enable_columnar_output
  cdef bool g_enable_union
  cdef bool g_enable_lazy_fetch
  cdef bool g_null_div_by_zero
  cdef bool g_enable_watchdog
  cdef bool g_enable_dynamic_watchdog
  cdef bool g_enable_debug_timer

cdef extern from "omniscidb/Logger/Logger.h" namespace "logger":
  cdef cppclass CLogOptions "logger::LogOptions":
    CLogOptions(const char*)

  cdef void CInitLogger "logger::init"(const CLogOptions &)
