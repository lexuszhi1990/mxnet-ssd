### C++

dependencies:
    - mxnet 1.2.0 or above
    - opencv 3.0 or above

1. build mxnet
    build mxnet from source refered to [site](https://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=Python&processor=CPU)

2. build app

```
cd c++
mkdir build && cd build
make clean && make && ./ssd-pedestrian ../../demo_images/street.jpg
```

### Python api

the easist way to install mxnet is [pip](https://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=Python&processor=CPU)

then

```
cd python
python3.6 main.py ../samples/ebike-three.jpg
```
