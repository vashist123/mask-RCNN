# Mask R-CNN Project

## Running Instructions

To run the code, it's pretty simple. 

Just use `python3 dataset.py` and `python3 rpn.py`. It assumes the `data` folder exists in the same directory as the Python files. 

To run the RPN `python3 main_rpn_train.py` and `python3 resume_rpn_train.py` and `python3 main_rpn_infer.py`. It assumes the `data` folder exists in the same directory as the Python files.

No external dependencies were used other than the ones we normally use in the class (NumPy, PyTorch, SciPy, etc.).


## Behavior of Files

* Both `dataset.py` and `rpn.py` have main functions which one-by-one produce plots required for the project report. After closing one image, the next image will be displayed.

* You can modify which checkpoint `resume_*_train.py` resumes (it's documented inside the file, pretty easy to change the checkpoint file name).

* Both `main_*_train.py` and `resume_*_train.py` will print training status and plot the 3 different loss curves at the end of training.

* You can modify which checkpoint `main_*_infer.py` loads (it's documented inside the file, pretty easy to change the checkpoint file name).

* `main_*_infer.py` will one by one-by-one show plot the top 20 proposals, and the proposals before and after NMS, for each image in the training set.

