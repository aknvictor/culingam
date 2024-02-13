To compile and run tests:


```bash
nvcc  -arch=sm_86 -rdc=true -o unit.so --shared -Xcompiler -fPIC ../tests/unit.cu  ../culingam/basic.cu -I../culingam/include -lnvToolsExt
python main.py
```
