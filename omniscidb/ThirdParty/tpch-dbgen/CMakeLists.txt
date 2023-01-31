add_executable(dbgen bcd2.c bm_utils.c build.c driver.c load_stub.c permute.c print.c rnd.c rng64.c speed_seed.c text.c)

target_compile_definitions(dbgen PRIVATE DBNAME="dss" RNG_TEST _FILE_OFFSET_BITS=64 TPCH EOL_HANDLING)
if(ENABLE_ASAN)
  target_compile_options(dbgen PRIVATE "-fno-sanitize=address")
  target_link_options(dbgen PRIVATE "-fno-sanitize=address")
endif()

if(NOT WIN32)
  target_compile_definitions(dbgen PRIVATE LINUX)
endif()
