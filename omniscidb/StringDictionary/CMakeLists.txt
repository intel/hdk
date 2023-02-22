add_library(StringDictionary StringDictionary.cpp StringDictionaryProxy.cpp)

target_link_libraries(StringDictionary OSDependent Utils ${Boost_LIBRARIES} ${PROFILER_LIBS} TBB::tbb)
if(ENABLE_FOLLY)
  target_link_libraries(StringDictionary Folly::folly)
endif()
