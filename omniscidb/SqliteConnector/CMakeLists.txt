add_library(SqliteConnector SqliteConnector.cpp SqliteConnector.h)

target_link_libraries(SqliteConnector PUBLIC SQLite::SQLite3 PRIVATE Logger ${Boost_THREAD_LIBRARY})
