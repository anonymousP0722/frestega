

## Setup Instructions

Follow these steps to get the project running on your system:

### Install Dependencies
Run the following command to install the required packages:
```shell
python -m pip install -r requirements.txt
```

### Build Cython Files
Compile the Cython files using:
```shell
python baselines/setup.py build_ext --inplace
```

### Run the Application
To execute the program, use:
```shell
python frestega/frestega.py --secret_bits_file bit_stream.txt --gpuid 1 --generate_num 10
```
