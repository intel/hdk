add_library(SqliteConnector SqliteConnector.cpp SqliteConnector.h)

target_link_libraries(SqliteConnector PUBLIC sqlite3 PRIVATE Logger ${Boost_THREAD_LIBRARY})
