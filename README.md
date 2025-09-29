# PPT-AI
Anything that is trained with pytorch

## Dependencies
- Linux: Tested with Ubuntu 24.04
- Docker: https://docs.docker.com/engine/install/ubuntu/ & https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user
- Nvidia container toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- VS Code: https://code.visualstudio.com/ (Extensions: Python, Dev Containers)
- Workstation with Nvidia GPU

## Get into the Container
In VS Code press ctrl+shift+P and search for "Dev Containers: Rebuild and Reopen in Container". The first time this will take a bit as Docker as to pull the Image. Once its ready you will be in the container and you will see only the workspace folder. Switch back with ctrl+shift+P and "Dev Containers: Reopen folder localy".

## Run and Debug Code


## Container Usage
#### Container already in use
```log
Error response from daemon: Conflict. The container name "/ppt-ai-dev" is already in use by container "fbb94ded6b754f6629d3fb197e05f32d9cafd73f97ade0032e2edaa440da25db". You have to remove (or rename) that container to be able to reuse that name.
```
In case something crashed it can happen that the dev container has not shut down. In this case you have to manually stop it: `docker stop ppt-ai-dev && docker rm ppt-ai-dev`

#### File permissions
If files are generated inside the container they are owned by root. To not have any pain with them editing outside you can do this outside of the container:
`sudo chown -R $(whoami) .`

#### GPU util
You can test where your bottlenecks are by setting the run_profiler=True. Or your GPU util with `watch -n 1 nvidia-smi`