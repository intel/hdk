cd %~dp0\..\..\..

cmake -B build -S . %*
cmake --build build --config Release --parallel

