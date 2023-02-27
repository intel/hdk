#include "Shared/funcannotations.h"
#define SHARED_EXPORT RUNTIME_EXPORT

#include "threading.h"
#if DISABLE_CONCURRENCY
#elif ENABLE_TBB
namespace threading_tbb {
SHARED_EXPORT ::tbb::task_arena g_tbb_arena;
}
#endif
