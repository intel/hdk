cd %~dp0\..\..\..

rem Set PATH variable to all subdirectories where DLLs are located
rem Make sure you don't add spaces in the lines below (even at the
rem beginning of the line) because it adds spaces to paths.

set PATH=%PATH%;%CONDA_PREFIX%\Library\lib\jvm\bin\server
set WORKDIR=%CD%\build\omniscidb
set MODE=Release
set PATH=%PATH%;^
%WORKDIR%\ThirdParty\googletest\%MODE%;^
%WORKDIR%\Tests\ArrowSQLRunner\%MODE%;^
%WORKDIR%\ArrowStorage\%MODE%;^
%WORKDIR%\IR\%MODE%;^
%WORKDIR%\QueryEngine\%MODE%;^
%WORKDIR%\Logger\%MODE%;^
%WORKDIR%\Shared\%MODE%;^
%WORKDIR%\SqliteConnector\%MODE%;^
%WORKDIR%\StringDictionary\%MODE%;^
%WORKDIR%\DataMgr\%MODE%;^
%WORKDIR%\Calcite\%MODE%;^
%WORKDIR%\Analyzer\%MODE%;^
%WORKDIR%\Utils\%MODE%;^
%WORKDIR%\SchemaMgr\%MODE%;^
%WORKDIR%\L0Mgr\%MODE%;^
%WORKDIR%\OSDependent\Windows\%MODE%;^
%WORKDIR%\CudaMgr\%MODE%;^
%WORKDIR%\ResultSet\%MODE%;^
%WORKDIR%\ResultSetRegistry\%MODE%;^
%WORKDIR%\QueryBuilder\%MODE%;^
%WORKDIR%\QueryOptimizer\%MODE%;^
%WORKDIR%\ConfigBuilder\%MODE%

set PATH

ldd build\omniscidb\Tests\Release\ArrowBasedExecuteTest.exe

cmake --build build --config Release --target sanity_tests --verbose
