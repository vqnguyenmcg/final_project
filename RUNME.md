### To build:

The container can be built using the following command from within this directory:

```
docker build -t image1 ./ 
```

### To run:

The software can be run from within this directory with the following command:

```
docker run -ti -v ${PWD}/data:/data -v ${PWD}/figures:/figures image1 /data /figures
```
