## Docker

## Unofficial Dockerfile for 3D Gaussian Splatting for Real-Time Radiance Field Rendering
## Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis
## https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

# Use the base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

# NOTE:
# Building the libraries for this repository requires cuda *DURING BUILD PHASE*, therefore:
# - The default-runtime for container should be set to "nvidia" in the deamon.json file. See this: https://github.com/NVIDIA/nvidia-docker/issues/1033
# - For the above to work, the nvidia-container-runtime should be installed in your host. Tested with version 1.14.0-rc.2
# - Make sure NVIDIA's drivers are updated in the host machine. Tested with 525.125.06

ENV DEBIAN_FRONTEND=noninteractive



ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"

COPY ./environment.yml /tmp/environment.yml
WORKDIR /tmp/

RUN conda env create --file environment.yml
RUN conda init bash && exec bash && conda activate vggt

RUN apt update && apt install -y git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1

COPY ./requirements.txt /tmp/requirements.txt
RUN conda run -n vggt pip install -r requirements.txt

COPY ./requirements_demo.txt /tmp/requirements_demo.txt
RUN conda run -n vggt pip install -r requirements_demo.txt


WORKDIR /vggt/

# COPY environment.yml /tmp/environment.yml
# COPY gaussian_splatting/ /tmp/gaussian_splatting/
# WORKDIR /tmp/
# RUN conda env create --file environment.yml
# RUN rm /tmp/environment.yml

# COPY submodules/AtomGS /tmp/AtomGS
# WORKDIR /tmp/AtomGS/
# RUN conda env create --file environment.yml

# RUN  conda init bash && exec bash && conda activate gaussian_splatting

# # Install colmap
# RUN apt update && apt-get install -y \
#     git \
#     cmake \
#     ninja-build \
#     build-essential \
#     libboost-program-options-dev \
#     libboost-filesystem-dev \
#     libboost-graph-dev \
#     libboost-system-dev \
#     libeigen3-dev \
#     libflann-dev \
#     libfreeimage-dev \
#     libmetis-dev \
#     libgoogle-glog-dev \
#     libgtest-dev \
#     libsqlite3-dev \
#     libglew-dev \
#     qtbase5-dev \
#     libqt5opengl5-dev \
#     libcgal-dev \
#     libceres-dev \
#     libomp-dev

# WORKDIR /tmp/
# RUN git clone https://github.com/colmap/colmap.git
# # COPY colmap/ /tmp/colmap/
# WORKDIR /tmp/colmap
# # Back up commit: 98940342171e27fbf7a52223a39b5b3f699f23b8
# RUN git checkout 682ea9ac4020a143047758739259b3ff04dabe8d &&\
#     mkdir build && cd build &&\
#     cmake .. -GNinja \
#     -DCMAKE_CUDA_ARCHITECTURES=all-major \
#     -DOPENMP_ENABLED=ON && \
#     ninja &&\
#     ninja install

# # WORKDIR /tmp/colmap/pycolmap
# # # Install pycolmap with OpenMP flags explicitly set
# # RUN conda run -n sugar bash -c "\
# #     export CMAKE_ARGS='-DOpenMP_C_FLAGS=-fopenmp \
# #                         -DOpenMP_C_LIB_NAMES=gomp \
# #                         -DOpenMP_gomp_LIBRARY=/usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so \
# #                         -DOpenMP_CXX_FLAGS=-fopenmp \
# #                         -DOpenMP_CXX_LIB_NAMES=gomp' && \
# #     python -m pip install . --retries 30 --timeout 360"

# RUN conda run -n sugar python -m pip install pycolmap

# # Install Node.js 21.x at the system level
# RUN curl -fsSL https://deb.nodesource.com/setup_21.x | bash - && \
#     apt update && apt-get install -y \
#     nodejs \
#     aptitude

# RUN aptitude install -y npm

# # Ensure the system Node.js takes priority over Conda's Node.js
# RUN echo 'export PATH=/usr/bin:$PATH' >> /etc/profile.d/system_node.sh && \
#     chmod +x /etc/profile.d/system_node.sh

# WORKDIR /sugar

# RUN pip3 install google-api-python-client google-auth google-auth-oauthlib watchdog

# # Default conda project
# RUN echo "conda activate sugar" >> ~/.bashrc

# # This error occurs because there’s a conflict between the threading layer used
# # by Intel MKL (Math Kernel Library) and the libgomp library, 
# # which is typically used by OpenMP for parallel processing. 
# # This often happens when libraries like NumPy or SciPy are used in combination
# # with a multithreaded application (e.g., your Docker container or Python environment).
# # Solution, set threading layer explicitly! (GNU or INTEL)
# ENV MKL_THREADING_LAYER=GNU

# # Set up Meshroom paths
# RUN echo "export ALICEVISION_ROOT=/sugar/submodules/Meshroom-2023.3.0-linux/Meshroom-2023.3.0/aliceVision/" >> ~/.bashrc &&\
#     echo "export PATH=$PATH:/sugar/submodules/Meshroom-2023.3.0-linux/Meshroom-2023.3.0/aliceVision/bin/" >> ~/.bashrc &&\
#     echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sugar/submodules/Meshroom-2023.3.0-linux/Meshroom-2023.3.0/aliceVision/lib/" >> ~/.bashrc &&\
#     echo 'export PATH=/usr/bin:$PATH' >> ~/.bashrc
