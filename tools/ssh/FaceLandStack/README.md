# Usage
- Set PATH
```
export PATH=/path/to/landstack/bin:$PATH
```
- Set PYTHONPATH
```
export PYTHONPATH=/path/to/landstack:$PYTHONPATH
```
- Mount data
```
selfctl mount _research_facelm/Isilon-datashare
selfctl mount _research_facelm/Isilon-modelshare
```

# Example in Python
```
from landstack.utils import misc
d = misc.load_pickle('/unsullied/sharefs/g:research_facelm/Isilon-datashare/baseimg/landmark_alltest.info')
```

# Benchmark
Using builtin `lsk-test` command will invoke different benchmark test script.
 ```
 lsk-test general.accuracy /unsullied/sharefs/_research_facelm/Isilon-modelshare/model_zoo/float.demo --keys black-valid
 lsk-test general.accuracy /unsullied/sharefs/_research_facelm/Isilon-modelshare/model_zoo/lowbit.demo --keys black-valid --on-mobile
 ```
 This command will evaluate `float.demo` and `lowbit.demo` on `general.accuracy` benchmark, which locates `/path/to/landstack/benchmark/general/accuracy.py`.
 
 ```
 
------------------------------------------------------------
TestUnit: general.accuracy
                all  contour     eye  eyebrow    nose   mouth
black-valid 0.06604  0.08213 0.04932  0.07864 0.05269 0.06201
------------------------------------------------------------
 ```
 
 If want to known the accepted args of `general.accuracy`, please try `lsk-test general.accuracy -zz`. Accutually this will trigger an error and prompt the help.

# Others
- Please track your own libs in `own_libs` folder, followed by own name. For example, `own_libs/chenxi/misc.py`